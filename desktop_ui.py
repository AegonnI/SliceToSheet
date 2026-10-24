"""Qt desktop and VTK viewport for SliceToSheet."""
import datetime
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import traceback
from dataclasses import asdict
from types import SimpleNamespace

os.environ.setdefault('QT_API', 'pyside6')
import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor
from model_preview import preview_mesh, prepare_preview, source_key, transform_preview, preview_extents
from slicer import Settings, run, slicing_plan
from pipeline import save_config, load_config, Paused

DOCUMENTATION_URL = 'https://github.com/AegonnI/SliceToSheet/blob/main/docs/PARAMETERS_RU.md'
MODEL_FILTER = 'Модели STL / STEP (*.stl *.STL *.step *.STEP *.stp *.STP);;Все файлы (*)'
_font_path = Path(os.environ.get('WINDIR', r'C:\Windows'))/'Fonts'/'segoeui.ttf'
VIEWPORT_FONT = str(_font_path) if _font_path.is_file() else None
_application = None


def application():
    global _application
    _application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    _application.setApplicationName('SliceToSheet')
    _application.setOrganizationName('SliceToSheet')
    return _application


class NumberInput(QtWidgets.QDoubleSpinBox):
    def __init__(self, value, minimum=0, maximum=10000000, decimals=2):
        super().__init__()
        self.setRange(minimum, maximum)
        self.setDecimals(decimals)
        self.setValue(value)
        self.setKeyboardTracking(False)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class ChoiceInput(QtWidgets.QComboBox):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class Section(QtWidgets.QWidget):
    def __init__(self, title, expanded=True):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.toggle = QtWidgets.QToolButton(text=title, checkable=True, checked=expanded)
        self.toggle.setObjectName('sectionHeader')
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.body = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self.body)
        self.form.setContentsMargins(5, 2, 5, 10)
        self.form.setSpacing(9)
        self.form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addWidget(self.toggle)
        layout.addWidget(self.body)
        self.toggle.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)

    def set_expanded(self, expanded):
        self.body.setVisible(expanded)
        self.toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow)


class Worker(QtCore.QThread):
    event = QtCore.Signal(str, object)

    def __init__(self, task, parent):
        super().__init__(parent)
        self.task = task

    def run(self):
        try:
            result = self.task(self.event.emit)
            if result:
                self.event.emit(*result)
        except Paused as error:
            self.event.emit('paused', str(error))
        except Exception as error:
            self.event.emit('error', (str(error), traceback.format_exc()))


class App(QtWidgets.QMainWindow):
    def __init__(self):
        application()
        super().__init__()
        self.setWindowTitle('SliceToSheet 1.1 — модель → слои и листы A4')
        self.resize(1440, 940)
        self.setMinimumSize(1040, 680)
        self.busy = False
        self.result = None
        self.current_output = None
        self.stop_event = threading.Event()
        self._worker = None
        self._closing = False
        self._loading_sizes = False
        self._pending_settings = None
        self._reload_after_job = False
        self._preview_requested = False
        self.preview_enabled = False
        self.model_data = None
        self.model_scale = 1.0
        self.fields, self.sections = {}, {}
        self.preferences = QtCore.QSettings()
        self._build_ui()
        self.preview_timer = QtCore.QTimer(self, singleShot=True, interval=160)
        self.preview_timer.timeout.connect(self.refresh_preview)
        self.apply_theme(self.preferences.value('dark_theme', True, type=bool))

    def section(self, key, title, expanded=True):
        group = Section(title, expanded)
        self.sections[key] = group
        self.settings_layout.addWidget(group)
        return group

    def number(self, group, key, label, value, minimum, maximum, decimals=2):
        widget = NumberInput(value, minimum, maximum, decimals)
        self.fields[key] = widget
        group.form.addRow(label, widget)
        if key in ('thickness', 'margin'):
            widget.valueChanged.connect(self.schedule_preview)
        return widget

    @staticmethod
    def note(text):
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setObjectName('muted')
        return label

    def _build_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        self.setCentralWidget(central)
        left = QtWidgets.QWidget()
        left.setMinimumWidth(365)
        left.setMaximumWidth(470)
        side = QtWidgets.QVBoxLayout(left)
        side.setContentsMargins(0, 0, 0, 0)
        title_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel('SliceToSheet')
        title.setObjectName('brand')
        title_row.addWidget(title)
        title_row.addStretch()
        self.theme_button = QtWidgets.QPushButton('Тема')
        self.theme_button.clicked.connect(lambda: self.apply_theme(not self.dark_theme))
        title_row.addWidget(self.theme_button)
        side.addLayout(title_row)
        side.addWidget(self.note('Послойная сборка · STL / STEP · миллиметры'))
        self.choose_button = QtWidgets.QPushButton('Открыть модель…')
        self.choose_button.clicked.connect(self.choose)
        side.addWidget(self.choose_button)
        self.path = QtWidgets.QLineEdit()
        self.path.setPlaceholderText('Файл STL, STEP или STP')
        self.path.editingFinished.connect(self.source_edited)
        side.addWidget(self.path)
        self.preview_button = QtWidgets.QPushButton('Предпросмотр модели')
        self.preview_button.setObjectName('secondaryAction')
        self.preview_button.clicked.connect(self.preview)
        side.addWidget(self.preview_button)
        self.scroll = QtWidgets.QScrollArea(widgetResizable=True)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.settings_widget = QtWidgets.QWidget()
        self.settings_layout = QtWidgets.QVBoxLayout(self.settings_widget)
        self.settings_layout.setContentsMargins(0, 4, 8, 4)
        self.settings_layout.setSpacing(5)
        self.scroll.setWidget(self.settings_widget)
        side.addWidget(self.scroll, 1)

        size = self.section('size', 'Размеры и нарезка')
        self.dimensions = {}
        for axis in 'XYZ':
            widget = NumberInput(.001, .001, decimals=3)
            widget.setEnabled(False)
            widget.setSuffix(' мм')
            widget.valueChanged.connect(lambda value, a=axis: self.dimension_changed(a, value))
            size.form.addRow(f'Габарит {axis}', widget)
            self.dimensions[axis] = widget
        self.size_note = self.note('Габариты связаны: пропорции модели сохраняются.')
        size.form.addRow(self.size_note)
        self.axis = ChoiceInput()
        self.axis.addItems(['X', 'Y', 'Z'])
        self.axis.currentTextChanged.connect(self.schedule_preview)
        size.form.addRow('Ось нарезки', self.axis)
        self.number(size, 'thickness', 'Толщина слоя, мм', 20, .1, 1000)
        size.form.addRow(self.note('Плоскость слоя перпендикулярна выбранной оси.'))

        holes = self.section('holes', 'Шпильки и отверстия')
        self.number(holes, 'diameter', 'Диаметр шпильки, мм', 8, .1, 100)
        self.number(holes, 'clearance', 'Зазор по диаметру, мм', .2, 0, 20)
        self.number(holes, 'wall', 'Материал до края, мм', 2, 0, 100)
        self.fields['wall'].setToolTip('От поверхности отверстия до границы материала: наружной поверхности, полости и краёв детали.')
        self.number(holes, 'min_hole_gap', 'Между отверстиями, мм', 8, 0, 1000)
        self.fields['min_hole_gap'].setToolTip('Минимальная перемычка между поверхностями всех отверстий одной детали, включая соединения вверх и вниз.')
        holes.form.addRow(self.note('До 2 соединений вверх и вниз. Длина шпильки рассчитывается по толщине слоёв.'))

        shell = self.section('shell', 'Оболочка и шпангоуты', False)
        self.make_shell = QtWidgets.QCheckBox('Создать полость в сплошной модели')
        shell.form.addRow(self.make_shell)
        self.number(shell, 'hull_wall', 'Толщина стенки, мм', 50, .1, 10000)
        self.frames = QtWidgets.QCheckBox('Добавлять шпангоуты')
        shell.form.addRow(self.frames)
        self.number(shell, 'frame_coefficient', 'Множитель стенки шпангоута', 2, 1.01, 20)
        self.number(shell, 'frame_interval', 'Интервал, слоёв', 25, 3, 10000, 0)
        shell.form.addRow(self.note('Шпангоут занимает слои N и N+1. Множитель 2 удваивает стенку внутрь. Для готовой оболочки укажите её фактическую стенку.'))

        export = self.section('export', 'Экспорт и отчёты')
        self.export_format = ChoiceInput()
        for label, value in [('SVG', 'svg'), ('DXF', 'dxf'), ('SVG и DXF', 'both')]:
            self.export_format.addItem(label, value)
        export.form.addRow('Чертежи', self.export_format)
        self.export_parts = QtWidgets.QCheckBox('Сохранять отдельные STL')
        self.export_parts.setChecked(True)
        self.export_assembly = QtWidgets.QCheckBox('Сохранять assembly.stl')
        self.export_assembly.setChecked(True)
        export.form.addRow(self.export_parts)
        export.form.addRow(self.export_assembly)
        export.form.addRow(self.note('Площади, периметры, карта сборки и база продолжения сохраняются независимо от STL.'))

        advanced = self.section('advanced', 'Дополнительные параметры', False)
        self.number(advanced, 'margin', 'Поля A4, мм', 5, 0, 89)
        self.orientation = ChoiceInput()
        for label, value in [('Автоматически', 'auto'), ('Вертикально · 210 × 297', 'portrait'), ('Горизонтально · 297 × 210', 'landscape')]:
            self.orientation.addItem(label, value)
        self.orientation.currentIndexChanged.connect(self.schedule_preview)
        advanced.form.addRow('Ориентация A4', self.orientation)
        self.number(advanced, 'svg_padding', 'Внешний отступ SVG, мм', 0, 0, 100)
        self.fields['svg_padding'].setToolTip('Увеличивает документ за пределы A4. Для A4 оставьте 0 и печатайте в масштабе 100%.')
        self.number(advanced, 'checkpoint_layers', 'Сохранять каждые N слоёв', 10, 1, 1000, 0)
        self.number(advanced, 'shell_pitch', 'Шаг поверхности полости, мм', 3, .05, 100)
        advanced.form.addRow(self.note('Меньший шаг полости повышает детализацию и расход памяти. Стенка задаётся после масштабирования.'))
        simplify = self.section('simplify', 'Упрощение тяжёлой модели', False)
        self.simplify = QtWidgets.QCheckBox('Упростить сетку для расчёта')
        simplify.form.addRow(self.simplify)
        self.number(simplify, 'simplify_faces', 'Целевое число треугольников', 200000, 100, 10000000, 0)
        self.number(simplify, 'simplify_tolerance', 'Допустимое отклонение, мм', .5, .001, 100, 3)
        simplify.form.addRow(self.note('Для просмотра всегда используется облегчённая копия. Упрощение для расчёта меняет геометрию и проверяется отдельно.'))
        self.settings_layout.addStretch()
        side.addWidget(QtWidgets.QLabel('Папка результатов'))
        destination_row = QtWidgets.QHBoxLayout()
        self.destination = QtWidgets.QLineEdit(str(Path.home()/'STL_A4_Results'))
        destination_row.addWidget(self.destination)
        folder = QtWidgets.QPushButton('…')
        folder.setFixedWidth(35)
        folder.clicked.connect(self.folder)
        destination_row.addWidget(folder)
        side.addLayout(destination_row)
        self.start_button = QtWidgets.QPushButton('Нарезать и экспортировать')
        self.start_button.setObjectName('primaryAction')
        self.start_button.clicked.connect(self.start)
        side.addWidget(self.start_button)
        self.pause_button = QtWidgets.QPushButton('Приостановить расчёт')
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.pause_job)
        side.addWidget(self.pause_button)
        more = QtWidgets.QHBoxLayout()
        self.resume_button = QtWidgets.QPushButton('Продолжить…')
        self.resume_button.clicked.connect(self.resume_job)
        self.config_menu = QtWidgets.QPushButton('Настройки ▾')
        menu = QtWidgets.QMenu(self.config_menu)
        menu.addAction('Сохранить настройки…', self.save_settings)
        menu.addAction('Загрузить настройки…', self.load_settings)
        menu.addAction('Открыть папку результата', self.open_results)
        menu.addAction('Локальная документация', self.open_local_documentation)
        self.config_menu.setMenu(menu)
        more.addWidget(self.resume_button)
        more.addWidget(self.config_menu)
        side.addLayout(more)
        docs = QtWidgets.QPushButton('Документация по параметрам ↗')
        docs.clicked.connect(self.open_documentation)
        side.addWidget(docs)
        layout.addWidget(left)
        right = QtWidgets.QVBoxLayout()
        toolbar = QtWidgets.QHBoxLayout()
        self.view_title = QtWidgets.QLabel('Предпросмотр модели')
        self.view_title.setObjectName('viewTitle')
        toolbar.addWidget(self.view_title)
        toolbar.addStretch()
        reset = QtWidgets.QPushButton('Вписать')
        reset.clicked.connect(lambda: self.plotter.reset_camera())
        toolbar.addWidget(reset)
        right.addLayout(toolbar)
        self.plotter = QtInteractor(self, auto_update=False, multi_samples=0)
        self.plotter.setMinimumSize(400, 300)
        self.plotter.add_axes()
        self.plotter.add_text('Откройте модель и включите предпросмотр', font_size=12,
                              font_file=VIEWPORT_FONT, color='#91a6bc', name='empty')
        right.addWidget(self.plotter.interactor, 1)
        self.model_details = self.note('Ось нарезки выбирается отдельно от размеров модели.')
        right.addWidget(self.model_details)
        layer_row = QtWidgets.QHBoxLayout()
        layer_row.addWidget(QtWidgets.QLabel('Готовый слой:'))
        self.layer_number = QtWidgets.QSpinBox(minimum=1, maximum=1000000)
        layer_row.addWidget(self.layer_number)
        self.layer_button = QtWidgets.QPushButton('Показать слой')
        self.layer_button.clicked.connect(self.show_layer)
        self.layer_button.setEnabled(False)
        layer_row.addWidget(self.layer_button)
        layer_row.addStretch()
        right.addLayout(layer_row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        right.addWidget(self.progress)
        self.status = QtWidgets.QLabel('Выберите файл модели. Все размеры — в миллиметрах.')
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(600)
        self.log.setMaximumHeight(145)
        right.addWidget(self.log)
        layout.addLayout(right, 1)

    def apply_theme(self, dark):
        self.dark_theme = bool(dark)
        self.preferences.setValue('dark_theme', self.dark_theme)
        colors = ('#171b23', '#202632', '#2b3442', '#edf1f7', '#a7b4c7', '#3b4659') if dark else ('#edf1f5', '#ffffff', '#e3e9f1', '#202c3d', '#55657b', '#c9d3df')
        background, panel, field, foreground, muted, border = colors
        self.setStyleSheet(f'''
            QWidget {{ color: {foreground}; font-family: "Segoe UI"; font-size: 12px; }}
            QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {background}; }}
            QLabel#brand {{ font-size: 25px; font-weight: 700; }}
            QLabel#viewTitle {{ font-size: 16px; font-weight: 600; }}
            QLabel#muted {{ color: {muted}; font-size: 11px; }}
            QToolButton#sectionHeader {{ background: {panel}; border: 1px solid {border}; border-radius: 6px; text-align: left; padding: 9px; font-weight: 600; }}
            QPushButton {{ background: {panel}; border: 1px solid {border}; border-radius: 5px; padding: 8px 10px; }}
            QPushButton:hover {{ border-color: #56a5c7; }}
            QPushButton:disabled {{ color: {muted}; }}
            QPushButton#primaryAction {{ background: #287ca0; color: white; font-weight: 600; padding: 11px; border: 0; }}
            QPushButton#secondaryAction {{ background: {field}; padding: 10px; }}
            QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit {{ background: {panel}; border: 1px solid {border}; border-radius: 4px; padding: 5px; selection-background-color: #287ca0; }}
            QComboBox QAbstractItemView, QMenu {{ background: {panel}; color: {foreground}; selection-background-color: #287ca0; }}
            QProgressBar {{ border: 1px solid {border}; border-radius: 3px; text-align: center; height: 13px; background: {panel}; }}
            QProgressBar::chunk {{ background: #3899b7; }}
            QScrollBar:vertical {{ background: {background}; width: 11px; margin: 0; }}
            QScrollBar::handle:vertical {{ background: {border}; border-radius: 5px; min-height: 30px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        ''')
        self.theme_button.setText('Светлая' if dark else 'Тёмная')
        self.plotter.set_background('#131a24' if dark else '#e6edf4')
        self.plotter.render()

    def choose(self):
        if not self.busy:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть модель', '', MODEL_FILTER)
            if path:
                self.set_source(path)

    def set_source(self, path):
        self.path.setText(str(path))
        self.model_data, self.model_scale, self.result = None, 1.0, None
        self.layer_button.setEnabled(False)
        self.load_source()

    def source_edited(self):
        if self.busy or not self.path.text().strip():
            return
        try:
            if self.model_data is None or source_key(self.path.text()) != self.model_data.source_key:
                self.set_source(self.path.text())
        except OSError:
            pass

    def load_source(self):
        source = self.path.text().strip()
        if not Path(source).is_file():
            self.show_error('Выберите существующий файл STL, STEP или STP.')
            return
        self.status.setText('Чтение модели и подготовка облегчённого просмотра…')
        self.worker(lambda emit: ('loaded', prepare_preview(source)), 'load')

    def dimension_changed(self, axis, value):
        if not self._loading_sizes and self.model_data is not None:
            self.model_scale = value/self.model_data.extents['XYZ'.index(axis)]
            self.update_dimensions()
            self.schedule_preview()

    def update_dimensions(self):
        if self.model_data is not None:
            self._loading_sizes = True
            try:
                for i, axis in enumerate('XYZ'):
                    self.dimensions[axis].setEnabled(True)
                    dimension = float(self.model_data.extents[i]*self.model_scale)
                    self.dimensions[axis].setMaximum(max(10000000, dimension*1.01))
                    self.dimensions[axis].setValue(dimension)
            finally:
                self._loading_sizes = False
            self.size_note.setText(f'Пропорции сохранены · масштаб ×{self.model_scale:.6g}')

    def schedule_preview(self, *_):
        if self.preview_enabled:
            self.preview_timer.start()

    def preview(self):
        self.preview_enabled = self._preview_requested = True
        if self.busy:
            return
        try:
            valid = self.model_data is not None and source_key(self.path.text()) == self.model_data.source_key
        except OSError:
            valid = False
        self.refresh_preview() if valid else self.load_source()

    def refresh_preview(self):
        if not self.preview_enabled or self.model_data is None:
            return
        try:
            cfg = self.cfg()
            mesh = transform_preview(self.model_data, self.model_scale, cfg.axis)
            exact_extents = preview_extents(self.model_data, self.model_scale, cfg.axis)
            steps, page, counts = slicing_plan(SimpleNamespace(extents=exact_extents), cfg)
            self.draw([mesh], f'Исходная модель · нарезка вдоль {cfg.axis}')
            self.model_details.setText(
                f'Треугольников: {self.model_data.source_faces:,} → {len(mesh.faces):,} в просмотре\n'
                f'После поворота: {exact_extents[0]:.1f} × {exact_extents[1]:.1f} × {exact_extents[2]:.1f} мм · '
                f'слоёв: {counts[2]} · A4: {page[0]:.0f} × {page[1]:.0f} мм · деталь до {steps[0]:.0f} × {steps[1]:.0f} мм')
            self.preview_button.setText('Обновить предпросмотр')
            self._preview_requested = False
        except ValueError as error:
            self.status.setText(str(error))

    def draw(self, meshes, title):
        self.plotter.clear()
        self.plotter.enable_lightkit()
        self.view_title.setText(title)
        colors, bounds = ['#62a9c6', '#dfb56c', '#83bb9f', '#b998cb', '#da9382'], []
        for index, mesh in enumerate(meshes):
            faces = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).ravel()
            self.plotter.add_mesh(pv.PolyData(np.asarray(mesh.vertices), faces), color=colors[index % len(colors)],
                                  show_edges=False, name=f'model_{index}', render=False)
            bounds.extend(mesh.bounds)
        if not bounds:
            self.plotter.render()
            return
        bounds = np.asarray(bounds)
        lo, hi = bounds.min(axis=0), bounds.max(axis=0)
        spans = np.maximum(hi-lo, 1)
        plane = pv.Plane(center=((lo[0]+hi[0])/2, (lo[1]+hi[1])/2, lo[2]-.002*max(spans)),
                         direction=(0, 0, 1), i_size=spans[0]*1.14, j_size=spans[1]*1.14,
                         i_resolution=10, j_resolution=10)
        self.plotter.add_mesh(plane, color='#50667c' if self.dark_theme else '#aebecd', opacity=.35,
                              show_edges=True, edge_color='#65798d', name='slicing_base', render=False)
        arrow_start = [hi[0]+.07*spans[0], lo[1], lo[2]]
        self.plotter.add_mesh(pv.Arrow(start=arrow_start, direction=(0, 0, 1), scale=max(spans[2]*.25, max(spans)*.08)),
                              color='#f0b657', name='slicing_direction', render=False)
        self.plotter.add_text(f'Плоскость слоя перпендикулярна {self.axis.currentText()} · ход нарезки вверх',
                              position='upper_left', font_size=10, font_file=VIEWPORT_FONT,
                              color='#bdcddd' if self.dark_theme else '#34465c', name='axis_caption')
        self.plotter.add_axes()
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    def cfg(self):
        values = {key: widget.value() for key, widget in self.fields.items()}
        for key in ('checkpoint_layers', 'frame_interval', 'simplify_faces'):
            values[key] = int(values[key])
        if not self.simplify.isChecked():
            values['simplify_faces'] = 0
        orientation = self.orientation.currentData()
        cfg = Settings(**values, axis=self.axis.currentText(), scale=self.model_scale, target_length=0,
                       length=2*values['thickness'], auto_dowel_length=True, auto_orientation=orientation == 'auto',
                       landscape=orientation == 'landscape', staggered=True, four_holes=True, unique_keys=False,
                       side_connections=False, reverse=False, frames=self.frames.isChecked(), make_shell=self.make_shell.isChecked(),
                       export_format=self.export_format.currentData(), export_parts_stl=self.export_parts.isChecked(),
                       export_assembly_stl=self.export_assembly.isChecked())
        cfg.validate()
        return cfg

    def apply_settings(self, cfg):
        self._pending_settings = cfg if self.model_data is None else None
        for key, value in asdict(cfg).items():
            if key in self.fields:
                self.fields[key].setValue(value)
        self.axis.setCurrentText(cfg.axis)
        self.orientation.setCurrentIndex(0 if cfg.auto_orientation else (2 if cfg.landscape else 1))
        self.frames.setChecked(cfg.frames)
        self.make_shell.setChecked(cfg.make_shell)
        self.simplify.setChecked(cfg.simplify_faces > 0)
        self.export_format.setCurrentIndex(max(0, self.export_format.findData(cfg.export_format)))
        self.export_parts.setChecked(cfg.export_parts_stl)
        self.export_assembly.setChecked(cfg.export_assembly_stl)
        self.model_scale = cfg.scale
        if self.model_data is not None and cfg.target_length:
            self.model_scale = cfg.target_length/self.model_data.extents['XYZ'.index(cfg.axis)]
        self.update_dimensions()
        self.schedule_preview()

    def save_settings(self):
        if not self.busy:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Сохранить настройки', '', 'Настройки (*.json)')
            if path:
                try:
                    save_config(path, self.cfg())
                except Exception as error:
                    self.show_error(str(error))

    def load_settings(self):
        if not self.busy:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Загрузить настройки', '', 'Настройки (*.json)')
            if path:
                try:
                    self.apply_settings(load_config(path))
                    self.append_log('Настройки загружены. Новые расчёты используют перевязку и четыре межслойных отверстия.')
                except Exception as error:
                    self.show_error(str(error))

    def folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Папка результатов', self.destination.text())
        if path:
            self.destination.setText(path)

    def start(self):
        if self.busy:
            return
        try:
            source = self.path.text().strip()
            if not Path(source).is_file():
                raise ValueError('Выберите файл модели.')
            if self.model_data is None or source_key(source) != self.model_data.source_key:
                raise ValueError('Дождитесь чтения выбранной модели.')
            cfg = self.cfg()
            output = Path(self.destination.text())/datetime.datetime.now().strftime('sliced_%Y%m%d_%H%M%S_%f')
        except Exception as error:
            self.show_error(str(error))
            return
        self.launch_job(source, output, cfg)

    def launch_job(self, source, output, cfg, resume=False):
        self.current_output, self.result = Path(output), None
        self.layer_button.setEnabled(False)
        self.append_log('Папка расчёта: '+str(output))
        def task(emit):
            report = run(source, output, cfg, log=lambda line: emit('log', line), resume=resume,
                         stop=self.stop_event, progress=lambda s, v, t: emit('progress', (s, v, t)))
            return 'done', (report, Path(output))
        self.worker(task, 'job')

    def resume_job(self):
        if self.busy:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Расчёт с progress.sqlite')
        if not folder:
            return
        try:
            db_path = Path(folder)/'progress.sqlite'
            if not db_path.is_file():
                raise ValueError('В выбранной папке нет progress.sqlite.')
            with sqlite3.connect(str(db_path)) as db:
                values = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM meta')}
            cfg = Settings.from_saved(values['settings'])
            cfg.validate()
            source = values['source']
            if not Path(source).is_file():
                source, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Укажите исходную модель', '', MODEL_FILTER)
                if not source:
                    return
            self.path.setText(source)
            if self.model_data is not None and source_key(source) != self.model_data.source_key:
                self.model_data = None
                for widget in self.dimensions.values():
                    widget.setEnabled(False)
            self._reload_after_job = self.model_data is None
            self.apply_settings(cfg)
            self.launch_job(source, Path(folder), cfg, resume=True)
        except Exception as error:
            self.show_error(str(error))

    def show_layer(self):
        if self.busy or self.result is None:
            return
        number, output = self.layer_number.value(), self.result[1]
        def task(emit):
            from pipeline import load_layer_meshes
            meshes = load_layer_meshes(output, number)
            budget = max(100, 100000//max(1, len(meshes)))
            return 'layer', ([preview_mesh(m, budget) for m in meshes], number)
        self.worker(task, 'layer')

    def worker(self, task, kind):
        if self.busy:
            return
        self.busy, self.worker_kind = True, kind
        self.stop_event.clear()
        for widget in (self.start_button, self.choose_button, self.resume_button, self.config_menu, self.path):
            widget.setEnabled(False)
        self.settings_widget.setEnabled(kind != 'job')
        self.pause_button.setEnabled(kind == 'job')
        self.progress.setRange(0, 0)
        self._worker = Worker(task, self)
        self._worker.event.connect(self.handle_event)
        self._worker.finished.connect(self.task_idle)
        self._worker.start()

    @QtCore.Slot(str, object)
    def handle_event(self, event, value):
        if event == 'loaded':
            self.model_data = value
            if self._pending_settings is not None:
                cfg, self._pending_settings = self._pending_settings, None
                self.apply_settings(cfg)
            else:
                self.update_dimensions()
            self.status.setText(f'Модель прочитана: {value.source_faces:,} треугольников.')
            if self.preview_enabled:
                self.refresh_preview()
        elif event == 'log':
            self.append_log(value)
        elif event == 'progress':
            stage, current, total = value
            self.progress.setRange(0, max(1, total))
            self.progress.setValue(current)
            self.status.setText(f'{stage}: {current}/{total}')
        elif event == 'layer':
            meshes, number = value
            self.draw(meshes, f'Готовый слой {number}')
        elif event == 'done':
            self.result = value
            report, output = value
            self.layer_number.setMaximum(report['layer_count'])
            self.layer_button.setEnabled(True)
            totals = report.get('metrics', {}).get('totals', {})
            text = f'Готово · деталей: {len(report["parts"])} · шпилек: {len(report["connections"])} · замечаний: {len(report["warnings"])}'
            if totals:
                text += f'\nМатериал сечений: {totals.get("net_area_mm2", 0)/1e6:.4f} м² · рез: {totals.get("total_cut_mm", 0)/1000:.2f} м'
            self.append_log(text+'\n'+str(output))
            if report['warnings']:
                self.log.appendPlainText('Подробности недостающих соединений и маркировки — в REPORT.txt.')
        elif event == 'paused':
            self.append_log(value+'\nПапка для продолжения: '+str(self.current_output))
        elif event == 'error':
            self.append_log(value[1])
            self.show_error(value[0])

    @QtCore.Slot()
    def task_idle(self):
        self.busy = False
        for widget in (self.start_button, self.choose_button, self.resume_button, self.config_menu, self.path, self.settings_widget):
            widget.setEnabled(True)
        self.pause_button.setEnabled(False)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()
        if self._closing:
            self.close()
        elif self._reload_after_job:
            self._reload_after_job = False
            self.load_source()
        elif self._preview_requested and self.model_data is not None:
            self.refresh_preview()

    def pause_job(self):
        self.stop_event.set()
        self.status.setText('Пауза после текущей геометрической операции…')

    def append_log(self, line):
        self.log.appendPlainText(str(line))
        self.status.setText(str(line))

    def show_error(self, message):
        QtWidgets.QMessageBox.critical(self, 'SliceToSheet', message)

    def open_documentation(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(DOCUMENTATION_URL))

    def open_local_documentation(self):
        root = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(root/'docs'/'PARAMETERS_RU.md')))

    def open_results(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.current_output or Path(self.destination.text()))))

    def closeEvent(self, event):
        if self.busy:
            self._closing = True
            self.pause_job()
            event.ignore()
        else:
            self.plotter.close()
            event.accept()

    def mainloop(self):
        self.show()
        return application().exec()
