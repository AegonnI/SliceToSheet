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
        from app import App, preview_mesh, application

        window = App()
        window.show()
        application().processEvents()
        preview = preview_mesh(trimesh.creation.icosphere(subdivisions=3), 300)
        window.draw([preview], 'SliceToSheet installation check')
        window.plotter.render()
        application().processEvents()
        window.plotter.screenshot(str(Path(destination).with_suffix('.png')))
        result['renderer'] = window.plotter.ren_win.GetClassName()
        window.close()
        application().processEvents()
        result['checks'].append('Qt desktop, VTK 3D viewport and mesh simplification')

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
            report = run(source, output, Settings(export_format='both'), log=lambda _: None)
            if report['layer_count'] != 2 or len(report['parts']) != 2:
                raise RuntimeError('Unexpected slicing result')
            if not list((output / 'sections_a4').rglob('*.svg')):
                raise RuntimeError('SVG export missing')
            if not list((output / 'sections_a4').rglob('*.dxf')):
                raise RuntimeError('DXF export missing')
            if not (output/'Inspection'/'index.html').is_file() or not (output/'print_addresses.csv').is_file():
                raise RuntimeError('Inspection or compact numbering index missing')
            if not report['sheet_nesting']['sheets'] or not list((output/'Sheets'/'DXF').glob('*.dxf')):
                raise RuntimeError('True-contour nesting or its native dependencies failed')
            result['checks'].append('Compact numbering, inspection index and true-contour A4 nesting')
            if not (output / 'metrics.csv').is_file() or report['metrics']['totals']['net_area_mm2'] <= 0:
                raise RuntimeError('Manufacturing measurements missing')
            assembly = trimesh.load(output / 'assembly.stl', force='mesh', process=False)
            # Touching parts stay separate in the assembly; welding their
            # coincident vertices would create artificial non-manifold edges.
            np.testing.assert_allclose(assembly.volume, report['volume_after_holes'], rtol=1e-5)
            for part in (output / 'stl').glob('*.stl'):
                if not trimesh.load(part, force='mesh').is_volume:
                    raise RuntimeError('Invalid part STL: ' + part.name)
            run(source, output, resume=True, log=lambda _: None)
            result['checks'].append('STL import, slicing, dowels, STL/SVG/DXF export, measurements and resume')
            from pipeline import load_layer_meshes
            no_stl = root/'without_stl'
            run(stl, no_stl, Settings(export_format='dxf', export_parts_stl=False,
                                    export_assembly_stl=False), log=lambda _: None)
            if (no_stl/'stl').exists() or (no_stl/'assembly.stl').exists() or not load_layer_meshes(no_stl, 1):
                raise RuntimeError('Independent exports or database layer viewer failed')
            result['checks'].append('Optional STL exports and database layer preview')
            import manifold3d as md
            from slicer import from_solid
            shell = root/'shell.stl'
            outer = md.Manifold.cube([80,80,100])
            cavity = md.Manifold.cube([60,60,80]).translate([10,10,10])
            from_solid(outer-cavity).export(shell)
            framed = run(shell, root/'frames', Settings(frames=True, hull_wall=10,
                         frame_interval=3, frame_coefficient=1.5, export_parts_stl=False,
                         export_assembly_stl=False), log=lambda _: None)
            if framed['frames']['added_volume_mm3'] <= 0:
                raise RuntimeError('Frame geometry failed')
            result['checks'].append('Native Open3D/scikit-image frame geometry')
            stiffened = run(shell, root/'stringers', Settings(stringers=True, stringer_spacing=50,
                            stringer_depth=10, stringer_height=8, frames=True, hull_wall=10,
                            frame_interval=3, frame_coefficient=1.5, frame_layers=1, stringer_up='-X', export_parts_stl=False,
                            export_assembly_stl=False), log=lambda _: None)
            if stiffened['stringers']['added_volume_mm3'] <= 0 or stiffened['frames']['added_volume_mm3'] <= 0:
                raise RuntimeError('Combined stringer/frame geometry failed')
            result['checks'].append('Longitudinal stringers combined with transverse frames')
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
    sys.exit(App().mainloop())
