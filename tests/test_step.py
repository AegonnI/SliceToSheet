import builtins
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from OCP.BRep import BRep_Builder
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.TopoDS import TopoDS_Compound

from slicer import Settings, load_model, run
from pipeline import Paused


class StepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_step(self, shape, name='model.step'):
        path = self.root / name
        writer = STEPControl_Writer()
        self.assertEqual(writer.Transfer(shape, STEPControl_AsIs), IFSelect_RetDone)
        self.assertEqual(writer.Write(str(path)), IFSelect_RetDone)
        return path

    def test_extensions_units_and_transforms_match_stl(self):
        stl = self.root / 'box.stl'
        box = trimesh.creation.box([20, 30, 40])
        box.apply_translation([10, 15, 20])
        box.export(stl)
        for cfg in (Settings(), Settings(axis='X', reverse=True, target_length=100)):
            expected, transform = load_model(stl, cfg)
            for suffix in ('.step', '.stp', '.STEP', '.StP'):
                source = self.write_step(BRepPrimAPI_MakeBox(20, 30, 40).Shape(), 'box'+suffix)
                actual, actual_transform = load_model(source, cfg)
                np.testing.assert_allclose(actual.bounds, expected.bounds, atol=1e-6)
                np.testing.assert_allclose(actual_transform, transform, atol=1e-6)
                self.assertAlmostEqual(actual.volume, expected.volume, places=4)
        # An equivalent file whose source coordinates are measured in metres.
        source = self.write_step(BRepPrimAPI_MakeBox(.02, .03, .04).Shape())
        text = source.read_text().replace('SI_UNIT(.MILLI.,.METRE.)', 'SI_UNIT($,.METRE.)')
        self.assertIn('SI_UNIT($,.METRE.)', text)
        source.write_text(text)
        mesh, _ = load_model(source, Settings())
        np.testing.assert_allclose(mesh.extents, [20, 30, 40], atol=1e-6)

    def test_curved_and_positioned_multiple_bodies(self):
        shape = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(shape)
        builder.Add(shape, BRepPrimAPI_MakeCylinder(10, 20).Shape())
        builder.Add(shape, BRepPrimAPI_MakeBox(gp_Pnt(50, 0, 0), 20, 30, 40).Shape())
        source = self.write_step(shape)
        unicode_path = self.root / 'детали.stp'
        source.rename(unicode_path)
        mesh, _ = load_model(unicode_path, Settings())
        self.assertTrue(mesh.is_volume)
        self.assertEqual(len(mesh.split()), 2)
        np.testing.assert_allclose(mesh.extents, [80, 40, 40], atol=.05)
        self.assertAlmostEqual(mesh.volume, 24000 + np.pi*100*20, delta=15)

    def test_invalid_and_open_step(self):
        source = self.root / 'bad.step'
        source.write_text('not a STEP file')
        with self.assertRaisesRegex(ValueError, 'STEP'):
            load_model(source, Settings())
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCP.gp import gp_Pln, gp_Dir
        face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0,0,0), gp_Dir(0,0,1)), 0, 20, 0, 30).Face()
        source = self.write_step(face)
        with self.assertRaisesRegex(ValueError, 'негерметичен'):
            load_model(source, Settings())

    def test_stl_without_ocp_and_missing_dependency_message(self):
        original_import = builtins.__import__
        def without_ocp(name, *args, **kwargs):
            if name == 'OCP' or name.startswith('OCP.'):
                raise ImportError('OCP unavailable')
            return original_import(name, *args, **kwargs)
        path = self.root / 'box.stl'
        trimesh.creation.box().export(path)
        with patch('builtins.__import__', side_effect=without_ocp):
            self.assertTrue(load_model(path, Settings())[0].is_volume)
            with self.assertRaisesRegex(ValueError, 'cadquery-ocp'):
                load_model(self.root / 'box.step', Settings())

    def test_step_slice_export_and_resume(self):
        source = self.write_step(BRepPrimAPI_MakeBox(40, 40, 40).Shape())
        stop = threading.Event()
        output = self.root / 'output'
        def progress(stage, n, total):
            if stage == 'Нарезка':
                stop.set()
        with self.assertRaises(Paused):
            run(source, output, Settings(), log=lambda _: None, stop=stop, progress=progress)
        report = run(source, output, resume=True, log=lambda _: None)
        self.assertEqual(report['layer_count'], 2)
        self.assertEqual(len(report['parts']), 2)
        assembly = trimesh.load(output / 'assembly.stl', force='mesh')
        self.assertAlmostEqual(assembly.volume, report['volume_after_holes'], places=2)
        for part in (output / 'stl').glob('*.stl'):
            self.assertTrue(trimesh.load(part, force='mesh').is_volume)
        self.assertTrue(list((output / 'sections_a4').glob('*.svg')))


if __name__ == '__main__':
    unittest.main()
