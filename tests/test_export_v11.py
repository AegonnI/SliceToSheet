import csv
import json
import tempfile
import threading
import unittest
from itertools import product
from pathlib import Path

import numpy as np
import trimesh

from pipeline import Paused, load_layer_meshes
from slicer import Settings, from_solid, run


class ExportV11Tests(unittest.TestCase):
    def test_format_and_stl_exports_are_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'box.stl'
            trimesh.creation.box([70, 50, 15]).export(source)
            for fmt, pieces, assembly in product(('svg', 'dxf', 'both'), (False, True), (False, True)):
                with self.subTest(format=fmt, pieces=pieces, assembly=assembly):
                    out = root/f'{fmt}_{pieces}_{assembly}'
                    cfg = Settings(export_format=fmt, export_parts_stl=pieces,
                                   export_assembly_stl=assembly)
                    report = run(source, out, cfg, log=lambda _: None)
                    self.assertEqual((out/'stl').exists(), pieces)
                    self.assertEqual((out/'assembly.stl').exists(), assembly)
                    for extension in ('svg', 'dxf'):
                        expected = len(report['parts']) if fmt in (extension, 'both') else 0
                        self.assertEqual(len(list((out/'sections_a4').rglob('*.'+extension))), expected)
                    meshes = load_layer_meshes(out, 1)
                    self.assertEqual(len(meshes), len(report['parts']))
                    self.assertAlmostEqual(sum(m.volume for m in meshes), report['volume_after_holes'])
                    self.assertTrue(all(m.metadata['part_id'] for m in meshes))
                    if assembly:
                        combined = trimesh.load(out/'assembly.stl', force='mesh')
                        self.assertAlmostEqual(combined.volume, report['volume_after_holes'], places=2)

    def test_net_section_area_cut_lengths_and_surface_are_distinct(self):
        import ezdxf
        import manifold3d as md
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'ring.stl'
            solid = md.Manifold.cube([100, 80, 20]) - md.Manifold.cube([60, 40, 40]).translate([20, 20, -10])
            from_solid(solid).export(source)
            report = run(source, root/'result', Settings(export_format='both'), log=lambda _: None)
            totals = report['metrics']['totals']
            for key, expected in dict(net_area_mm2=5600, outer_cut_mm=360, inner_cut_mm=200,
                                      total_cut_mm=560, open_cut_mm=0, surface_area_mm2=22400).items():
                self.assertAlmostEqual(totals[key], expected, places=5)
            self.assertEqual(totals['inner_contours'], 1)
            self.assertEqual(report['metrics']['layers'][0]['net_area_mm2'], 5600)
            dxf = ezdxf.readfile(next((root/'result'/'sections_a4').rglob('*.dxf')))
            self.assertEqual(dxf.units, 4)
            self.assertFalse(dxf.audit().has_errors)
            contours = [e for e in dxf.modelspace() if e.dxf.layer == 'CONTOURS']
            self.assertEqual(len(contours), 2)
            self.assertTrue(all(e.closed for e in contours))
            lengths = []
            for entity in contours:
                points = np.array([(p[0], p[1]) for p in entity.get_points()])
                lengths.append(np.linalg.norm(points-np.roll(points, 1, axis=0), axis=1).sum())
            self.assertAlmostEqual(sum(lengths), totals['total_cut_mm'], places=6)
            texts = [e for e in dxf.modelspace() if e.dxftype() == 'TEXT']
            self.assertTrue(texts)
            self.assertTrue(all(e.dxf.rotation == 0 for e in texts))
            self.assertIn(report['parts'][0]['marking'], [e.dxf.text for e in texts])
            with (root/'result'/'metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual([r['scope'] for r in csv.DictReader(stream)], ['part', 'layer', 'total'])

    def test_resume_repairs_missing_selected_export_without_stl(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'box.stl'
            trimesh.creation.box([80, 60, 40]).export(source)
            cfg = Settings(export_format='both', export_parts_stl=False, export_assembly_stl=False)
            expected = run(source, root/'result', cfg, log=lambda _: None)
            target = next((root/'result'/'sections_a4').rglob('*.dxf'))
            target.unlink()
            actual = run(source, root/'result', resume=True, log=lambda _: None)
            self.assertTrue(target.is_file())
            self.assertEqual(actual['metrics'], expected['metrics'])
            self.assertFalse((root/'result'/'stl').exists())
            self.assertFalse((root/'result'/'assembly.stl').exists())

    def test_pause_during_dxf_export_preserves_geometry_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'box.stl'
            trimesh.creation.box([80, 60, 60]).export(source)
            cfg = Settings(export_format='dxf', export_parts_stl=False, export_assembly_stl=False)
            expected = run(source, root/'baseline', cfg, log=lambda _: None)
            stop = threading.Event()
            def progress(stage, number, total):
                if stage == 'Экспорт' and number == 1:
                    stop.set()
            with self.assertRaises(Paused):
                run(source, root/'paused', cfg, log=lambda _: None, progress=progress, stop=stop)
            actual = run(source, root/'paused', resume=True, log=lambda _: None)
            self.assertEqual(actual['metrics'], expected['metrics'])
            self.assertEqual(actual['connections'], expected['connections'])
            self.assertTrue((root/'paused'/'metrics.csv').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
