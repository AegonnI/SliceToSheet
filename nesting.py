"""Bounded true-contour nesting; exact final distances, no scale or reflection."""
from copy import deepcopy
from dataclasses import dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.signal import fftconvolve
import shapely
from shapely import affinity
from shapely.geometry import Polygon
PACKER_REVISION = 2


@dataclass
class Item:
    pid: str
    shape: object
    root: object
    source_height: float
    digest: str


def exports_present(folder,summary,cfg):
    if not cfg.nest_sheets:return True
    if not summary or not (Path(folder)/'Sheets'/'layout.json').is_file():return False
    formats=('svg','dxf') if cfg.export_format=='both' else (cfg.export_format,)
    return all((Path(folder)/'Sheets'/ext.upper()/f'Sheet_{number:05d}.{ext}').is_file()
               for number in range(1,summary['sheets']+1) for ext in formats)


def read_item(folder,pid,extension):
    from sections import section_path,even_odd
    path=section_path(folder,pid,extension)
    if not path.is_file():path=section_path(folder,pid,extension,True)
    content=path.read_bytes()
    polygons=[]
    if extension=='svg':
        root=ET.fromstring(content);height=float(root.get('height').removesuffix('mm'))
        def walk(node,tx=0.,ty=0.):
            if node.tag.rsplit('}',1)[-1] not in ('text','path') and node.get('transform'):
                transform=node.get('transform')
                if not transform.startswith('translate('):raise ValueError('Неизвестное преобразование контура.')
                values=[float(v) for v in re.findall(r'[-+]?(?:\d*\.\d+|\d+)',transform)]
                tx+=values[0];ty+=values[1] if len(values)>1 else 0
            if node.tag.rsplit('}',1)[-1]=='polyline':
                p=np.array([[float(v) for v in pair.split(',')] for pair in node.get('points').split()])
                p[:,0]+=tx;p[:,1]=height-p[:,1]-ty
                if len(p)<4 or np.linalg.norm(p[0]-p[-1])>1e-5:
                    raise ValueError(pid+': открытый контур нельзя включать в раскладку.')
                polygons.append(Polygon(p))
            for child in node:walk(child,tx,ty)
        walk(root)
    else:
        import ezdxf
        root=ezdxf.readfile(path);height=0.
        for entity in root.modelspace():
            if entity.dxf.layer=='CONTOURS':
                if not entity.closed:raise ValueError(pid+': открытый контур нельзя включать в раскладку.')
                polygons.append(Polygon([(p[0],p[1]) for p in entity.get_points()]))
    if any(not p.is_valid for p in polygons):raise ValueError(pid+': некорректный контур нельзя включать в раскладку.')
    shape=even_odd(polygons)
    return Item(pid,shape,root,height,hashlib.sha256(content).hexdigest())


class Packer:
    def __init__(self,page,edge,gap,cancel=lambda:None):
        self.page=np.array(page,dtype=float);self.edge=edge;self.gap=gap;self.cancel=cancel
        self.step=2.;self.target_gap=gap+.03
        self.nx,self.ny=np.ceil((self.page-2*edge+self.target_gap)/self.step).astype(int)
        self.boards=[];self.shapes=[];self.placements=[];self.heights=[]

    def raster(self,shape):
        x0,y0,x1,y1=shape.bounds
        nx=min(self.nx,int(math.ceil((x1-x0+self.target_gap)/self.step)))
        ny=min(self.ny,int(math.ceil((y1-y0+self.target_gap)/self.step)))
        x,y=np.meshgrid((np.arange(nx)+.5)*self.step-self.target_gap/2,
                         (np.arange(ny)+.5)*self.step-self.target_gap/2)
        return shapely.intersects_xy(shape.buffer(self.target_gap/2,quad_segs=8),x,y).astype(np.float32)

    def add_raster(self,board,shape):
        expanded=shape.buffer(self.target_gap/2,quad_segs=8)
        origin=self.edge-self.target_gap/2
        x0,y0,x1,y1=expanded.bounds
        left=max(0,int(math.floor((x0-origin)/self.step)));right=min(self.nx,int(math.ceil((x1-origin)/self.step)))
        bottom=max(0,int(math.floor((y0-origin)/self.step)));top=min(self.ny,int(math.ceil((y1-origin)/self.step)))
        if right<=left or top<=bottom:return
        x,y=np.meshgrid(origin+(np.arange(left,right)+.5)*self.step,origin+(np.arange(bottom,top)+.5)*self.step)
        board[bottom:top,left:right]=np.maximum(board[bottom:top,left:right],shapely.intersects_xy(expanded,x,y))

    def variants(self,item):
        result=[]
        rectangle=list(item.shape.minimum_rotated_rectangle.exterior.coords)
        a,b=max(zip(rectangle,rectangle[1:]),key=lambda pair:math.dist(*pair))
        aligned=(-math.degrees(math.atan2(b[1]-a[1],b[0]-a[0])))%180
        angles={0.,90.,180.,270.}
        angles.update(round((aligned+extra)%360,8) for extra in (0,90,180,270))
        for angle in sorted(angles):
            rotated=affinity.rotate(item.shape,angle,origin=(0,0))
            x0,y0,x1,y1=rotated.bounds;w,h=x1-x0,y1-y0
            if w>self.page[0]-2*self.edge+1e-6 or h>self.page[1]-2*self.edge+1e-6:continue
            shape=affinity.translate(rotated,-x0,-y0)
            result.append(dict(angle=angle,shape=shape,w=w,h=h,x0=x0,y0=y0,mask=self.raster(shape)))
        if not result:raise ValueError(item.pid+': деталь не помещается на выбранный лист 1:1.')
        return result

    def refine(self,v,x,y,existing):
        for _ in range(10):
            self.cancel()
            x=float(np.clip(x,self.edge,self.page[0]-self.edge-v['w']))
            y=float(np.clip(y,self.edge,self.page[1]-self.edge-v['h']))
            q=affinity.translate(v['shape'],x,y)
            if not existing:return x,y,q
            distances=shapely.distance(existing,q)
            j=int(np.argmin(distances));distance=float(distances[j])
            if distance>=self.target_gap-1e-7 and not np.any(shapely.intersects(existing,q)):return x,y,q
            from shapely.ops import nearest_points
            a,b=nearest_points(q,existing[j]);delta=np.array([a.x-b.x,a.y-b.y]);length=np.linalg.norm(delta)
            if length<1e-8:return None
            x,y=np.array([x,y])+delta/length*(self.target_gap-distance+1e-4)
        return None

    def append(self,item,placement):
        number=placement['sheet']-1
        while len(self.boards)<=number:
            self.boards.append(np.zeros((self.ny,self.nx),dtype=np.float32));self.shapes.append([]);self.heights.append(0.)
        q=affinity.translate(affinity.rotate(item.shape,placement['rotation'],origin=(0,0)),placement['tx'],placement['ty'])
        self.shapes[number].append(q);self.add_raster(self.boards[number],q);self.placements.append(placement)
        self.heights[number]=max(self.heights[number],q.bounds[3])

    def place(self,item):
        self.cancel();variants=self.variants(item)
        # Bounded best-fit pool avoids scanning thousands of previous sheets.
        need=min(v['mask'].sum() for v in variants)
        available=np.array([np.count_nonzero(b==0) for b in self.boards])
        eligible=np.flatnonzero(available>=need-.5)
        pages=list(dict.fromkeys([*eligible[np.argsort(available[eligible]-need)[:8]],*eligible[-4:]])) if len(eligible) else []
        for number in pages:
            proposals=[]
            for v in variants:
                self.cancel()
                collision=fftconvolve(self.boards[number],v['mask'][::-1,::-1],mode='valid')
                for flat in np.flatnonzero(collision<.25)[:30]:
                    ry,rx=divmod(int(flat),collision.shape[1]);x,y=self.edge+rx*self.step,self.edge+ry*self.step
                    if x+v['w']<=self.page[0]-self.edge+1e-6 and y+v['h']<=self.page[1]-self.edge+1e-6:
                        proposals.append((max(self.heights[number],y+v['h']),y,x,v['angle'],v))
            for _,y,x,_,v in sorted(proposals,key=lambda p:p[:4]):
                fit=self.refine(v,x,y,self.shapes[number])
                if fit is not None:
                    x,y,_=fit
                    result=dict(id=item.pid,sheet=int(number)+1,rotation=v['angle'],tx=x-v['x0'],ty=y-v['y0'])
                    self.append(item,result);return result
        v=min(variants,key=lambda v:(v['h'],v['w'],v['angle']))
        result=dict(id=item.pid,sheet=len(self.boards)+1,rotation=v['angle'],tx=self.edge-v['x0'],ty=self.edge-v['y0'])
        self.append(item,result);return result

    def verify(self):
        minimum=math.inf;edge=math.inf
        for shapes in self.shapes:
            self.cancel()
            for i,q in enumerate(shapes):
                x0,y0,x1,y1=q.bounds
                edge=min(edge,x0,y0,self.page[0]-x1,self.page[1]-y1)
                if i:
                    if np.any(shapely.intersects(shapes[:i],q)):raise ValueError('Пересечение деталей в раскладке.')
                    minimum=min(minimum,float(np.min(shapely.distance(shapes[:i],q))))
        if edge<self.edge-1e-5 or minimum<self.gap-1e-5:raise ValueError('Нарушены расстояния в раскладке.')
        return dict(minimum_edge_mm=None if math.isinf(edge) else edge,minimum_gap_mm=None if math.isinf(minimum) else minimum)


def export_sheet(number,placements,items,folder,page,cfg):
    import ezdxf
    from ezdxf.math import Matrix44
    from ezdxf.enums import TextEntityAlignment
    from sections import section_path
    ns='http://www.w3.org/2000/svg';ET.register_namespace('',ns)
    svg=ET.Element('{'+ns+'}svg',dict(width=f'{page[0]}mm',height=f'{page[1]}mm',viewBox=f'0 0 {page[0]} {page[1]}'))
    ET.SubElement(svg,'{'+ns+'}rect',{'width':'100%','height':'100%','fill':'white'})
    document=ezdxf.new('R2010');document.units=4;document.header['$MEASUREMENT']=1
    for name,color,plot in [('CONTOURS',7,1),('LABELS',2,1),('LEADERS',8,1),('SHEET_FRAME',9,0),('PERIMETER_INFO',3,0)]:
        document.layers.new(name,dxfattribs={'color':color,'plot':plot})
    document.styles.new('LABEL',dxfattribs={'font':'cour.ttf'});document.appids.new('BOAT_LABEL')
    formats=('svg','dxf') if cfg.export_format=='both' else (cfg.export_format,)
    for p in placements:
        item=items[p['id']];angle=math.radians(p['rotation']);cs,ss=math.cos(angle),math.sin(angle)
        if 'svg' in formats:
            source=item.root if item.source_height else ET.parse(section_path(folder/'sections_a4',item.pid,'svg')).getroot()
            height=item.source_height or float(source.get('height').removesuffix('mm'))
            group=ET.SubElement(svg,'{'+ns+'}g',{'data-part':item.pid,'transform':
                'matrix(%.12g %.12g %.12g %.12g %.12g %.12g)'%(cs,-ss,ss,cs,p['tx']-ss*height,page[1]-cs*height-p['ty'])})
            for child in source:
                if child.tag.rsplit('}',1)[-1]!='rect':group.append(deepcopy(child))
            for text in group.iter():
                if text.tag.rsplit('}',1)[-1]!='text' or not text.get('textLength'):continue
                transform=text.get('transform','')
                original=-float(re.findall(r'[-+]?(?:\d*\.\d+|\d+)',transform)[0]) if transform else 0.
                combined=(original+p['rotation']+180)%360-180
                if -90-1e-7<=combined<=90+1e-7:continue
                x,y=float(text.get('x')),float(text.get('y'));w=float(text.get('textLength'));h=float(text.get('font-size'))
                rad=math.radians(original)
                cx=x+math.cos(rad)*w/2-math.sin(rad)*h*.45
                cy=y-math.sin(rad)*w/2-math.cos(rad)*h*.45
                text.set('transform','rotate(180 %.12g %.12g) %s'%(cx,cy,transform))
        if 'dxf' in formats:
            source=ezdxf.readfile(section_path(folder/'sections_a4',item.pid,'dxf'))
            matrix=Matrix44.chain(Matrix44.z_rotate(angle),Matrix44.translate(p['tx'],p['ty'],0))
            for entity in source.modelspace():
                if entity.dxf.layer in ('CONTOURS','LABELS','LEADERS'):
                    copy=entity.copy();copy.transform(matrix)
                    if copy.dxftype()=='TEXT' and copy.dxf.layer=='LABELS' and copy.dxf.halign==5:
                        a=np.array(tuple(copy.dxf.insert)[:2]);b=np.array(tuple(copy.dxf.align_point)[:2]);v=b-a;length=np.linalg.norm(v)
                        if length:
                            angle_text=math.degrees(math.atan2(v[1],v[0]))
                            if angle_text>90+1e-7 or angle_text<-90-1e-7:
                                normal=np.array([-v[1],v[0]])/length
                                offset=normal*copy.dxf.height*.9
                                copy.set_placement(tuple(b+offset),tuple(a+offset),align=TextEntityAlignment.FIT)
                                copy.dxf.rotation=(angle_text+360)%360-180
                    document.modelspace().add_entity(copy)
    cut=math.fsum(items[p['id']].shape.length for p in placements)
    area=math.fsum(items[p['id']].shape.area for p in placements)
    for i,text in enumerate([f'SHEET {number:05d} | PARTS {len(placements)} | SCALE 1:1 | MM',
                            'CUT %.6f mm | NET AREA %.6f mm2'%(cut,area)]):
        document.modelspace().add_text(text,dxfattribs={'layer':'PERIMETER_INFO','height':2.5,'insert':(0,-8-i*4)})
    ET.SubElement(svg,'{'+ns+'}metadata').text=json.dumps(dict(sheet=number,source_parts=[p['id'] for p in placements],
        cut_mm=cut,net_area_mm2=area,scale='1:1',page_mm=page))
    document.modelspace().add_lwpolyline([(0,0),tuple([page[0],0]),tuple(page),tuple([0,page[1]])],close=True,dxfattribs={'layer':'SHEET_FRAME'})
    document.set_modelspace_vport(height=page[1],center=(page[0]/2,page[1]/2))
    for extension in formats:
        target=folder/'Sheets'/extension.upper()/f'Sheet_{number:05d}.{extension}'
        target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix(target.suffix+'.tmp')
        if extension=='svg':ET.ElementTree(svg).write(tmp,encoding='utf-8',xml_declaration=True)
        else:
            if document.audit().has_errors:raise ValueError('Ошибка DXF в раскладке.')
            document.saveas(tmp)
        os.replace(tmp,target)


def nest_job(folder,parts,page,cfg,cancel=lambda:None,log=print,notify=lambda *x:None):
    from pipeline import atomic_json
    folder=Path(folder);destination=folder/'Sheets';destination.mkdir(parents=True,exist_ok=True)
    extension='svg' if cfg.export_format in ('svg','both') else 'dxf'
    items={};excluded=[];digests=[]
    for i,part in enumerate(parts):
        cancel();item=read_item(folder/'sections_a4',part['id'],extension)
        digests.append((item.pid,hashlib.sha256(shapely.normalize(item.shape).wkb).hexdigest()))
        if item.shape.is_empty:excluded.append(item.pid)
        else:items[item.pid]=item
        if i%200==0:notify('Контуры раскладки',i+1,len(parts))
    signature=hashlib.sha256(json.dumps([PACKER_REVISION,page,cfg.margin,cfg.sheet_gap,sorted(digests)]).encode()).hexdigest()
    checkpoint=destination/'.packing_progress.json'
    packer=Packer(page,cfg.margin,cfg.sheet_gap,cancel)
    previous_path=checkpoint if checkpoint.is_file() else destination/'layout.json'
    if previous_path.is_file():
        previous=json.loads(previous_path.read_text(encoding='utf-8'))
        if 'summary' in previous:previous['signature']=previous['summary'].get('signature')
        if previous.get('signature')==signature:
            for p in previous['placements']:packer.append(items[p['id']],p)
    placed={p['id'] for p in packer.placements}
    order=sorted(items.values(),key=lambda item:(-(item.shape.bounds[2]-item.shape.bounds[0])*(item.shape.bounds[3]-item.shape.bounds[1]),item.pid))
    for item in order:
        if item.pid in placed:continue
        cancel();packer.place(item)
        if len(packer.placements)%100==0 or len(packer.placements)==len(items):
            atomic_json(checkpoint,dict(signature=signature,placements=packer.placements))
            log('Раскладка A4: %d/%d деталей, %d листов.'%(len(packer.placements),len(items),len(packer.boards)))
            notify('Раскладка A4',len(packer.placements),len(items))
    checks=packer.verify()
    if {p['id'] for p in packer.placements}!=set(items) or len(packer.placements)!=len(items):raise ValueError('Повтор или потеря деталей в раскладке.')
    for number in range(1,len(packer.boards)+1):
        cancel();export_sheet(number,[p for p in packer.placements if p['sheet']==number],items,folder,page,cfg)
        if number%25==0:notify('Листы раскладки',number,len(packer.boards))
    if set(items)|set(excluded)!={p['id'] for p in parts}:raise ValueError('Неполный индекс раскладки.')
    summary=dict(signature=signature,parts=len(items),source_parts=len(parts),excluded_empty_sections=excluded,
        sheets=len(packer.boards),page_mm=page,edge_mm=cfg.margin,
        required_gap_mm=cfg.sheet_gap,scale='1:1',rotation_degrees=sorted({p['rotation'] for p in packer.placements}),
        packer_revision=PACKER_REVISION,global_minimum_proven=False,
        method='True-contour raster search with exact continuous gap checks; each source part exactly once.',**checks)
    atomic_json(destination/'layout.json',dict(summary=summary,placements=packer.placements))
    extras=[]
    for ext in ('svg','dxf'):
        for path in (destination/ext.upper()).glob('Sheet_*.'+ext):
            if int(path.stem.split('_')[-1])>len(packer.boards):extras.append(path)
    if extras:
        archive=destination/'backups'/('previous_extra_sheets_'+str(time.time_ns()))
        for path in extras:
            target=archive/path.parent.name/path.name;target.parent.mkdir(parents=True,exist_ok=True);path.replace(target)
    with (destination/'placements.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['id','sheet','rotation','tx','ty']);writer.writeheader();writer.writerows(packer.placements)
    checkpoint.unlink(missing_ok=True)
    return summary
