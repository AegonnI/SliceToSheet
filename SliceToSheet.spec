"""Windows x64 portable folder, including the STEP native libraries."""
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

root = Path(SPECPATH)
data = collect_data_files('trimesh')
binaries = collect_dynamic_libs('OCP') + collect_dynamic_libs('shapely') + [item for item in collect_dynamic_libs('open3d')
                                       if 'torch_ops' not in item[0] and 'tf_ops' not in item[0]]
data += collect_data_files('open3d')
for distribution in ('numpy', 'trimesh', 'manifold3d', 'matplotlib', 'networkx',
                     'scipy', 'fast-simplification', 'cadquery-ocp', 'PySide6',
                     'pyvista', 'pyvistaqt', 'vtk', 'ezdxf', 'open3d', 'scikit-image', 'rtree', 'shapely'):
    data += copy_metadata(distribution)

a = Analysis(
    [str(root / 'launcher.py')], pathex=[str(root)], binaries=binaries, datas=data,
    # OCP 7.7.2 exposes its submodules dynamically from a single extension.
    hiddenimports=['scipy.spatial', 'scipy.sparse.csgraph', 'fast_simplification', 'OCP',
                   'vtkmodules.vtkRenderingOpenGL2', 'vtkmodules.vtkRenderingFreeType',
                   'vtkmodules.vtkInteractionStyle', 'skimage.measure', 'open3d.pybind'],
    hookspath=[], hooksconfig={'matplotlib': {'backends': ['Agg']}},
    runtime_hooks=[], excludes=['pytest', 'IPython', 'jupyter', 'notebook', 'PyQt5', 'PyQt6',
                                'PySide2', 'torch', 'tensorflow', 'open3d.ml.torch', 'open3d.ml.tf',
                                'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtPdf'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='SliceToSheet',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='SliceToSheet')
