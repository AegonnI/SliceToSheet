import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

from frames import FrameBuilder, is_frame_layer
from pipeline import Paused
from slicer import Settings, as_solid, cut, run


def box_shell(size, wall):
    outer = trimesh.creation.box(size)
    outer.apply_translation(np.asarray(size)/2)
    inner = trimesh.creation.box(np.asarray(size)-2*wall)
    inner.apply_translation(np.asarray(size)/2)
    inner.invert()
    return trimesh.util.concatenate([outer, inner])


class FrameV11Tests(unittest.TestCase):
    def test_single_layer_frame_and_legacy_default(self):
        mesh = box_shell([100, 90, 120], 10)
        cfg = Settings(frames=True, hull_wall=10, frame_coefficient=1.5, frame_interval=3,
                       frame_layers=1, thickness=21)
        cfg.validate()
        builder = FrameBuilder(mesh, cfg, log=lambda _: None)
        slab = as_solid(mesh).trim_by_plane([0, 0, 1], 42).trim_by_plane([0, 0, -1], -63)
        result = builder.augment(slab, 2)
        self.assertAlmostEqual(result.slice(50).area(), 100*90-70*60, delta=3)
        np.testing.assert_allclose(np.asarray(result.bounding_box()).reshape(2,3)[:,2], [42,63])
        self.assertTrue(is_frame_layer(2, cfg))
        self.assertFalse(is_frame_layer(3, cfg))
        self.assertEqual(Settings.from_saved({}).frame_layers, 2)
        Settings(frame_layers=3).validate()
        three = Settings(frames=True, frame_interval=5, frame_layers=3)
        self.assertEqual([k+1 for k in range(10) if is_frame_layer(k,three)], [5,6,7,10])
        for value in (0, 25, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Settings(frame_layers=value).validate()

    def test_frame_coefficient_is_total_thickness_ratio(self):
        mesh = box_shell([100, 90, 120], 10)
        cfg = Settings(frames=True, hull_wall=10, frame_coefficient=1.5, frame_interval=3)
        builder = FrameBuilder(mesh, cfg, log=lambda _: None)
        slab = as_solid(mesh).trim_by_plane([0, 0, 1], 40).trim_by_plane([0, 0, -1], -60)
        frame = builder.augment(slab, 2)
        self.assertAlmostEqual(frame.slice(50).area(), 100*90-70*60, delta=3)
        self.assertLess((slab-frame).volume(), 1e-4)
        self.assertFalse(is_frame_layer(1, cfg))
        self.assertTrue(is_frame_layer(2, cfg))
        self.assertTrue(is_frame_layer(3, cfg))

    def test_frame_accounting_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'shell.stl'
            box_shell([90, 80, 100], 10).export(source)
            cfg = Settings(frames=True, hull_wall=10, frame_coefficient=1.5, frame_interval=3,
                           export_parts_stl=False, export_assembly_stl=False, checkpoint_layers=2)
            expected = run(source, root/'baseline', cfg, log=lambda _: None)
            self.assertEqual(expected['frames']['layer_numbers'], [3, 4])
            self.assertEqual(expected['frames']['frame_wall_mm'], 15)
            self.assertAlmostEqual(expected['volume_before_holes'], box_shell([90, 80, 100], 10).volume+
                                   expected['frames']['added_volume_mm3'], places=2)
            stop = threading.Event()
            def progress(stage, number, total):
                if stage == 'Нарезка' and number == 3:
                    stop.set()
            with self.assertRaises(Paused):
                run(source, root/'paused', cfg, log=lambda _: None, progress=progress, stop=stop)
            actual = run(source, root/'paused', resume=True, log=lambda _: None)
            self.assertEqual(actual['frames'], expected['frames'])
            self.assertEqual(actual['metrics'], expected['metrics'])

    def test_shell_is_metric_and_cached_before_slicing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'solid.stl'
            trimesh.creation.box([80, 70, 100]).export(source)
            cfg = Settings(make_shell=True, hull_wall=10, shell_pitch=2, export_parts_stl=False,
                           export_assembly_stl=False, checkpoint_layers=2)
            stop = threading.Event()
            def progress(stage, number, total):
                if stage == 'Нарезка' and number == 1:
                    stop.set()
            with self.assertRaises(Paused):
                run(source, root/'paused', cfg, log=lambda _: None, progress=progress, stop=stop)
            self.assertTrue((root/'paused'/'.prepared_shell.npz').is_file())
            with patch('pipeline._make_shell', side_effect=AssertionError('must reuse exact prepared shell')):
                report = run(source, root/'paused', resume=True, log=lambda _: None)
            self.assertTrue(report['shell']['thickness_check']['passed'])
            self.assertEqual(report['shell']['wall_mm'], 10)
            self.assertLess(report['volume_before_holes'], 80*70*100)
            self.assertAlmostEqual(report['volume_before_holes'], report['shell']['shell_volume_mm3'], places=2)

    def test_existing_cavity_rejects_double_shell(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'shell.stl'
            box_shell([80, 70, 100], 10).export(source)
            with self.assertRaisesRegex(ValueError, 'уже содержит'):
                run(source, root/'result', Settings(make_shell=True, hull_wall=10), log=lambda _: None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
