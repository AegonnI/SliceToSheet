"""Russian desktop UI. Worker communicates with Tk only through a queue."""
import datetime
from pathlib import Path
import queue
import threading
import traceback
import json
import sqlite3
from dataclasses import asdict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh
from slicer import Settings, load_model, run, slicing_plan
from pipeline import save_config, load_config, Paused

MODEL_FILETYPES = [('Модели STL / STEP', '*.stl *.step *.stp'), ('STL', '*.stl'), ('STEP', '*.step *.stp'), ('Все файлы', '*.*')]

def preview_mesh(mesh, face_count=30000):
    """Simplify a copy by edge collapse, retaining a continuous surface."""
    if len(mesh.faces) <= face_count:
        return mesh
    return mesh.simplify_quadric_decimation(face_count=face_count)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('SliceToSheet — STL / STEP → слои / A4 / деревянные шпильки')
        self.geometry('1250x850');self.minsize(950,650)
        self.messages=queue.Queue();self.busy=False;self.result=None
        self.stop_event=threading.Event()
        self.current_output=None
        self.protocol('WM_DELETE_WINDOW',self.close_app)
        self.path=tk.StringVar();self.destination=tk.StringVar(value=str(Path.home()/'STL_A4_Results'))
        self.fields={};self.axis=tk.StringVar(value='X');self.landscape=tk.BooleanVar()
        self.reverse=tk.BooleanVar(value=True)
        self.staggered=tk.BooleanVar(value=True)
        self.unique_keys=tk.BooleanVar(value=False)
        self.four_holes=tk.BooleanVar(value=True)
        self.side_connections=tk.BooleanVar(value=False)
        holder=ttk.Frame(self);holder.pack(side='left',fill='y')
        scroller=tk.Canvas(holder,width=440,highlightthickness=0)
        scrollbar=ttk.Scrollbar(holder,orient='vertical',command=scroller.yview)
        scrollbar.pack(side='right',fill='y');scroller.pack(side='left',fill='y',expand=True)
        scroller.configure(yscrollcommand=scrollbar.set)
        sidebar=ttk.Frame(scroller,padding=12);scroller.create_window((0,0),window=sidebar,anchor='nw')
        sidebar.bind('<Configure>',lambda _:scroller.configure(scrollregion=scroller.bbox('all')))
        ttk.Label(sidebar,text='SliceToSheet',font=('Segoe UI',19,'bold')).pack(anchor='w')
        ttk.Label(sidebar,text='Размеры в миллиметрах. STL не хранит единицы.').pack(anchor='w',pady=(0,12))
        ttk.Button(sidebar,text='1. Выбрать STL / STEP',command=self.choose).pack(fill='x')
        ttk.Entry(sidebar,textvariable=self.path,width=44).pack(fill='x',pady=5)
        config_buttons=ttk.Frame(sidebar);config_buttons.pack(fill='x')
        ttk.Button(config_buttons,text='Сохранить настройки',command=self.save_settings).pack(side='left',expand=True,fill='x')
        ttk.Button(config_buttons,text='Загрузить настройки',command=self.load_settings).pack(side='left',expand=True,fill='x')
        form=ttk.Frame(sidebar);form.pack(fill='x',pady=8)
        settings=[('thickness','Слой, мм','20'),('diameter','Шпилька Ø, мм','8'),
                  ('length','Длина шпильки, мм','40'),('clearance','Зазор по диаметру, мм','0.2'),
                  ('wall','Запас стенки, мм','2'),('margin','Поля A4, мм','0'),
                  ('svg_padding','Свободное пространство SVG, мм','2'),
                  ('scale','Масштаб модели','1'),('connectors','Шпилек на стык (старый режим / боковые)','2'),
                  ('pitch','Шаг обычного поиска, мм','12'),
                  ('target_length','Длина вдоль оси, мм (0 = масштаб)','8000'),
                  ('key_tolerance','Допуск положения ключа, мм','0.5'),
                  ('checkpoint_layers','Сохранять каждые N слоёв','10')]
        for row,(key,label,value) in enumerate(settings):
            ttk.Label(form,text=label).grid(row=row,column=0,sticky='w',pady=3)
            var=tk.StringVar(value=value);self.fields[key]=var
            if key=='diameter':
                entry=ttk.Combobox(form,textvariable=var,values=['8','10'],state='readonly',width=10)
            else:
                entry=ttk.Entry(form,textvariable=var,width=12)
            entry.grid(row=row,column=1,sticky='e',padx=12)
        ttk.Label(form,text='Ось вдоль длины / хода нарезки').grid(row=len(settings),column=0,sticky='w')
        ttk.Combobox(form,textvariable=self.axis,values=['X','Y','Z'],state='readonly',width=10).grid(row=len(settings),column=1)
        ttk.Checkbutton(sidebar,text='Перевязка швов: смещение соседних слоёв',variable=self.staggered).pack(anchor='w')
        ttk.Checkbutton(sidebar,text='4 отверстия: 2 вниз + 2 вверх (тест)',variable=self.four_holes,command=self.select_four_holes).pack(anchor='w')
        ttk.Checkbutton(sidebar,text='Боковые шпильки внутри слоя',variable=self.side_connections).pack(anchor='w')
        ttk.Checkbutton(sidebar,text='Уникальные тройки отверстий (3 шпильки)',variable=self.unique_keys,command=self.select_unique_keys).pack(anchor='w')
        ttk.Label(sidebar,text='Проверка полных троек, допуск задаётся выше.\nНе исключает все ошибки сборки; см. REPORT.txt.').pack(anchor='w')
        ttk.Checkbutton(sidebar,text='Обратное направление: от максимума оси к минимуму',variable=self.reverse).pack(anchor='w')
        ttk.Label(sidebar,text='Заданная длина заменяет масштаб, пропорции сохраняются.\nСлои перпендикулярны выбранной исходной оси.').pack(anchor='w',pady=4)
        ttk.Checkbutton(sidebar,text='A4 горизонтально (297 × 210)',variable=self.landscape).pack(anchor='w')
        ttk.Label(sidebar,text='При полях 5 мм детали до 200 × 287 мм.\nДля полных 210 × 297 мм поставьте поля 0.').pack(anchor='w',pady=5)
        ttk.Label(sidebar,text='Отступ SVG добавляется с каждой стороны,\nне меняет размеры деталей и нарезку.\nПри 2 мм документ: 214 × 301 мм (портрет).\nПечать: 100% / фактический размер.\nДля A4 оставьте место настройкой «Поля A4»\nили печатайте на большем листе; не вписывайте.',wraplength=410).pack(anchor='w',pady=5)
        ttk.Button(sidebar,text='Предпросмотр исходной модели',command=self.preview).pack(fill='x',pady=5)
        ttk.Button(sidebar,text='2. Папка результатов',command=self.folder).pack(fill='x')
        ttk.Entry(sidebar,textvariable=self.destination).pack(fill='x',pady=5)
        self.start_button=ttk.Button(sidebar,text='3. Нарезать и экспортировать',command=self.start)
        self.start_button.pack(fill='x',pady=8)
        ttk.Button(sidebar,text='Продолжить сохранённый расчёт',command=self.resume_job).pack(fill='x')
        self.pause_button=ttk.Button(sidebar,text='Приостановить расчёт',command=self.pause_job,state='disabled')
        self.pause_button.pack(fill='x',pady=4)
        self.progress=ttk.Progressbar(sidebar,mode='indeterminate');self.progress.pack(fill='x')
        ttk.Label(sidebar,text='40 мм через два слоя по 20 мм — насквозь.\nДля скрытого стыка уменьшите длину.\nТонкие участки могут остаться без шпилек.\nЧитайте REPORT.txt перед изготовлением.',foreground='#8b4c10').pack(anchor='w',pady=10)
        main=ttk.Frame(self,padding=8);main.pack(side='right',fill='both',expand=True)
        self.figure=Figure(figsize=(7,5));self.axes=self.figure.add_subplot(111,projection='3d')
        self.canvas=FigureCanvasTkAgg(self.figure,master=main);self.canvas.get_tk_widget().pack(fill='both',expand=True)
        NavigationToolbar2Tk(self.canvas,main)
        layerbar=ttk.Frame(main);layerbar.pack(fill='x')
        ttk.Label(layerbar,text='Просмотр готового слоя:').pack(side='left')
        self.layer_number=tk.StringVar(value='1')
        ttk.Entry(layerbar,textvariable=self.layer_number,width=8).pack(side='left')
        ttk.Button(layerbar,text='Показать слой',command=self.show_layer).pack(side='left')
        self.status=tk.StringVar(value='Выберите STL / STEP. Сохраняются STL, SVG сечений и карта соединений.')
        ttk.Label(main,textvariable=self.status,wraplength=640).pack(fill='x',pady=5)
        self.log=tk.Text(main,height=10,wrap='word',state='disabled');self.log.pack(fill='x')
        self.after(100,self.poll)

    def select_four_holes(self):
        if self.four_holes.get(): self.unique_keys.set(False)

    def select_unique_keys(self):
        if self.unique_keys.get():
            self.four_holes.set(False)
            self.fields['connectors'].set('3')

    def apply_settings(self,c):
        for key,value in asdict(c).items():
            if key in self.fields:self.fields[key].set(str(value))
        self.axis.set(c.axis);self.reverse.set(c.reverse);self.landscape.set(c.landscape)
        self.staggered.set(c.staggered);self.unique_keys.set(c.unique_keys)
        self.four_holes.set(c.four_holes);self.side_connections.set(c.side_connections)

    def save_settings(self):
        if self.busy:return
        try:
            c=self.cfg();p=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('Настройки','*.json')])
            if p:save_config(p,c)
        except Exception as e:messagebox.showerror('Настройки',str(e))

    def load_settings(self):
        if self.busy:return
        p=filedialog.askopenfilename(filetypes=[('Настройки','*.json')])
        if p:
            try:self.apply_settings(load_config(p))
            except Exception as e:messagebox.showerror('Настройки',str(e))

    def pause_job(self):
        self.stop_event.set();self.status.set('Остановка после текущей геометрической операции…')

    def show_layer(self):
        if self.busy or self.result is None:return
        try:
            number=int(self.layer_number.get());report,output=self.result
            if not 1<=number<=report['layer_count']:raise ValueError('Номер слоя вне диапазона.')
            selected=[p for p in report['parts'] if p['layer_number']==number]
        except Exception as e:messagebox.showerror('Просмотр слоя',str(e));return
        def task():
            meshes=[]
            for p in selected:
                m=trimesh.load(output/'stl'/(p['id']+'.stl'),force='mesh');m.apply_translation(p['assembly_offset'])
                meshes.append(preview_mesh(m,max(12,60000//max(1,len(selected)))))
            if meshes:self.messages.put(('layer',(meshes,number)))
            else:self.messages.put(('log','Этот слой пуст.'))
        self.worker(task)

    def close_app(self):
        if self.busy:
            self.pause_job()
            self.after(200,self.close_app)
        else:self.destroy()

    def resume_job(self):
        if self.busy:return
        folder=filedialog.askdirectory(title='Папка расчёта с progress.sqlite')
        if not folder:return
        try:
            dbpath=Path(folder)/'progress.sqlite'
            if not dbpath.is_file():raise ValueError('Сохранённый расчёт не найден.')
            with sqlite3.connect(str(dbpath)) as db:
                values={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM meta')}
            c=Settings.from_saved(values['settings']);c.validate();source=values['source']
            if not Path(source).is_file():
                source=filedialog.askopenfilename(title='Укажите исходную модель для продолжения',filetypes=MODEL_FILETYPES)
                if not source:return
            self.apply_settings(c);self.path.set(source)
            self.launch_job(source,Path(folder),c,resume=True)
        except Exception as e:messagebox.showerror('Продолжение',str(e))

    def cfg(self):
        values={k:float(v.get().replace(',','.')) for k,v in self.fields.items()}
        if not values['connectors'].is_integer(): raise ValueError('Количество шпилек должно быть целым.')
        values['connectors']=int(values['connectors'])
        if not values['checkpoint_layers'].is_integer(): raise ValueError('Интервал сохранения должен быть целым.')
        values['checkpoint_layers']=int(values['checkpoint_layers'])
        c=Settings(**values,axis=self.axis.get(),landscape=self.landscape.get(),reverse=self.reverse.get(),
                   staggered=self.staggered.get(),unique_keys=self.unique_keys.get(),
                   four_holes=self.four_holes.get(),side_connections=self.side_connections.get());c.validate();return c

    def choose(self):
        p=filedialog.askopenfilename(filetypes=MODEL_FILETYPES)
        if p:self.path.set(p)

    def folder(self):
        p=filedialog.askdirectory()
        if p:self.destination.set(p)

    def draw(self, meshes, title):
        self.axes.clear()
        bounds=[]
        for index,mesh in enumerate(meshes):
            faces=mesh.triangles
            colors=['#4d91ad','#e6ad50','#8bbb86','#bd8cbb','#e3806d']
            self.axes.add_collection3d(Poly3DCollection(faces,facecolors=colors[index%5],edgecolor='none',alpha=1,shade=True))
            bounds.extend(mesh.bounds)
        limits=np.array(bounds);lo=limits.min(axis=0);hi=limits.max(axis=0)
        for setter,a,b in zip((self.axes.set_xlim,self.axes.set_ylim,self.axes.set_zlim),lo,hi):setter(a,b)
        self.axes.set_box_aspect(np.maximum(hi-lo,1))
        self.axes.set_xlabel('X, mm');self.axes.set_ylabel('Y, mm');self.axes.set_zlabel('Z, mm')
        self.axes.set_title(title);self.canvas.draw_idle()

    def worker(self, task):
        if self.busy:return
        self.busy=True;self.start_button.configure(state='disabled');self.progress.configure(mode='indeterminate');self.progress.start()
        self.stop_event.clear();self.pause_button.configure(state='normal')
        def wrapped():
            try:task()
            except Paused as e:self.messages.put(('log',str(e)+'\nПапка расчёта: '+str(self.current_output)))
            except Exception as e:
                self.messages.put(('log',traceback.format_exc()))
                self.messages.put(('error',str(e)))
            finally:self.messages.put(('idle',None))
        threading.Thread(target=wrapped,daemon=True).start()

    def preview(self):
        if self.busy:return
        try:c=self.cfg();source=self.path.get()
        except Exception as e:messagebox.showerror('Настройки',str(e));return
        def task():
            mesh,_=load_model(source,c)
            steps,_,counts=slicing_plan(mesh,c)
            details=('После поворота: %.1f × %.1f × %.1f мм. Слоёв: %d по %.1f мм. '
                     'Деталь до %.1f × %.1f мм. Z на экране — ход нарезки, исходная ось %s.'
                     % (*mesh.extents,counts[2],c.thickness,*steps[:2],c.axis))
            self.messages.put(('preview',(preview_mesh(mesh),details)))
        self.worker(task)

    def start(self):
        if self.busy:return
        try:
            c=self.cfg();source=self.path.get()
            if not Path(source).is_file():raise ValueError('Выберите файл STL, STEP или STP.')
            output=Path(self.destination.get())/datetime.datetime.now().strftime('sliced_%Y%m%d_%H%M%S_%f')
        except Exception as e:messagebox.showerror('Настройки',str(e));return
        self.launch_job(source,output,c)

    def launch_job(self,source,output,c,resume=False):
        self.current_output=output
        self.status.set('Папка расчёта: '+str(output))
        self.messages.put(('log','Папка для продолжения расчёта: '+str(output)))
        def task():
            report=run(source,output,c,lambda line:self.messages.put(('log',line)),resume=resume,
                       stop=self.stop_event,progress=lambda s,v,t:self.messages.put(('progress',(s,v,t))))
            # Do not reload thousands of output meshes merely to display completion.
            self.messages.put(('done',(report,output)))
        self.worker(task)

    def poll(self):
        try:
            while True:
                event,value=self.messages.get_nowait()
                if event=='log':
                    self.status.set(value);self.log.configure(state='normal');self.log.insert('end',value+'\n')
                    self.log.see('end');self.log.configure(state='disabled')
                    if int(self.log.index('end-1c').split('.')[0])>600:
                        self.log.configure(state='normal');self.log.delete('1.0','101.0');self.log.configure(state='disabled')
                elif event=='progress':
                    stage,current,total=value;self.progress.stop();self.progress.configure(mode='determinate',maximum=max(1,total),value=current)
                    self.status.set('%s: %d/%d'%(stage,current,total))
                elif event=='preview':self.draw([value[0]],'Исходная модель');self.status.set(value[1])
                elif event=='layer':self.draw(value[0],'Слой %d'%value[1])
                elif event=='done':
                    report,output=value
                    self.result=(report,output)
                    self.status.set(str(output))
                    text='Деталей: %d\nШпилек: %d\nЗамечаний: %d\n\n%s' % (len(report['parts']),len(report['connections']),len(report['warnings']),output)
                    if report['warnings']:messagebox.showwarning('Готово — проверьте REPORT.txt',text)
                    else:messagebox.showinfo('Готово',text)
                elif event=='error':messagebox.showerror('Не удалось обработать модель',value)
                elif event=='idle':self.busy=False;self.start_button.configure(state='normal');self.pause_button.configure(state='disabled');self.progress.stop()
        except queue.Empty:pass
        self.after(100,self.poll)

if __name__=='__main__':App().mainloop()
