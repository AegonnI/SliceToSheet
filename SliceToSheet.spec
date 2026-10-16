"""Windows x64 portable folder, including the STEP native libraries."""
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

root = Path(SPECPATH)
data = collect_data_files('trimesh')
binaries = []
for distribution in ('numpy', 'trimesh', 'manifold3d', 'matplotlib', 'networkx',
                     'scipy', 'fast-simplification', 'cadquery-ocp'):
    data += copy_metadata(distribution)

a = Analysis(
    [str(root / 'launcher.py')], pathex=[str(root)], binaries=binaries, datas=data,
    # OCP 7.7.2 exposes its submodules dynamically from a single extension.
    hiddenimports=['scipy.spatial', 'scipy.sparse.csgraph', 'fast_simplification', 'OCP'],
    hookspath=[], hooksconfig={'matplotlib': {'backends': ['TkAgg']}},
    runtime_hooks=[], excludes=['pytest', 'IPython', 'jupyter', 'notebook', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='SliceToSheet',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='SliceToSheet')
