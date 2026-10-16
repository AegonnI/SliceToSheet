"""Desktop entry point and installation diagnostics for the portable build."""
import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path


def check_installation(destination):
    """Check bundled native libraries, the desktop, STEP import and export."""
    result = {'ok': False, 'checks': []}
    try:
        import numpy as np
        import trimesh
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from slicer import Settings, load_model, run
        from app import App, preview_mesh

        window = App()
        window.withdraw()
        window.update_idletasks()
        preview = preview_mesh(trimesh.creation.icosphere(subdivisions=3), 300)
        window.draw([preview], 'SliceToSheet installation check')
        window.canvas.draw()
        window.destroy()
        result['checks'].append('Tk desktop, Matplotlib 3D and mesh simplification')

        with tempfile.TemporaryDirectory(prefix='SliceToSheet_') as temp:
            root = Path(temp)
            source = root / 'model.step'
            writer = STEPControl_Writer()
            if writer.Transfer(BRepPrimAPI_MakeBox(40, 40, 40).Shape(), STEPControl_AsIs) != IFSelect_RetDone:
                raise RuntimeError('STEP transfer failed')
            if writer.Write(str(source)) != IFSelect_RetDone:
                raise RuntimeError('STEP write failed')
            mesh, _ = load_model(source, Settings())
            np.testing.assert_allclose(mesh.extents, [40, 40, 40])
            result['checks'].append('STEP import and native CAD libraries')
            stl = root / 'model.stl'
            mesh.export(stl)
            load_model(stl, Settings())
            output = root / 'output'
            report = run(source, output, Settings(), log=lambda _: None)
            if report['layer_count'] != 2 or len(report['parts']) != 2:
                raise RuntimeError('Unexpected slicing result')
            if not list((output / 'sections_a4').glob('*.svg')):
                raise RuntimeError('SVG export missing')
            assembly = trimesh.load(output / 'assembly.stl', force='mesh', process=False)
            # Touching parts stay separate in the assembly; welding their
            # coincident vertices would create artificial non-manifold edges.
            np.testing.assert_allclose(assembly.volume, report['volume_after_holes'], rtol=1e-5)
            for part in (output / 'stl').glob('*.stl'):
                if not trimesh.load(part, force='mesh').is_volume:
                    raise RuntimeError('Invalid part STL: ' + part.name)
            run(source, output, resume=True, log=lambda _: None)
            result['checks'].append('STL import, slicing, dowels, STL/SVG export and resume')
        result['ok'] = True
    except Exception:
        result['error'] = traceback.format_exc()
    Path(destination).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SliceToSheet portable desktop application')
    parser.add_argument('--check-installation', metavar='REPORT_JSON', help='Check this installation and save a diagnostic report')
    args = parser.parse_args()
    if args.check_installation:
        sys.exit(check_installation(args.check_installation))
    from app import App
    App().mainloop()
