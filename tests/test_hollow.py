import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from hollow import (make_inner, prepare_mesh, shell_from_inner, validate_inner,
                    stl_safe_inner, run)
from slicer import Settings, load_model, as_solid


class HollowTests(unittest.TestCase):
    def test_saved_blocks_resume_without_distance_queries(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            mesh = trimesh.creation.box([80, 60, 50])
            cache = Path(folder) / 'blocks'
            first, _ = make_inner(mesh, 10, 2, 8, log=lambda _: None, cache_dir=cache)
            with patch('hollow.distance', side_effect=AssertionError('Cached block recomputed')):
                second, _ = make_inner(mesh, 10, 2, 8, log=lambda _: None, cache_dir=cache)
            self.assertAlmostEqual(first.volume, second.volume)

    def test_touching_cavities_become_representable_in_stl(self):
        a = trimesh.creation.box([2, 2, 2])
        b = a.copy(); b.apply_translation([2, 2, 0])
        native = trimesh.util.concatenate([a, b])
        result = stl_safe_inner(native, log=lambda _: None)
        self.assertTrue(result.is_volume)
        self.assertTrue(result.is_watertight)
        self.assertAlmostEqual(result.volume, 16, delta=.2)

    def test_full_export_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'box.stl'
            trimesh.creation.box([200, 160, 120]).export(path)
            report = run(path, root / 'result', length=200, wall=10, pitch=2,
                         tolerance=1.5, log=lambda _: None)
            self.assertTrue(report['thickness_check']['passed'])
            shell = trimesh.load(root / 'result' / 'hull_hollow.stl', force='mesh')
            self.assertTrue(shell.is_volume)

    def test_box_offset_and_cavity_survives_slicer(self):
        outer = trimesh.creation.box([200, 160, 120])
        inner, scene = make_inner(outer, wall=10, pitch=2, coarse_pitch=8, log=lambda _: None)
        check = validate_inner(inner, scene, 10, tolerance=1.5, samples=5000)
        self.assertTrue(check['passed'])
        np.testing.assert_allclose(inner.extents, [180, 140, 100], atol=.01)
        shell = shell_from_inner(outer, inner)
        expected_volume = 200*160*120 - 180*140*100
        self.assertAlmostEqual(shell.volume / expected_volume, 1., delta=.005)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'shell.stl'
            shell.export(path)
            loaded, _ = load_model(path, Settings(axis='X', target_length=200))
            self.assertAlmostEqual(loaded.volume / shell.volume, 1., places=5)
            solid = as_solid(loaded)
            # Along the middle of the hull, cross section must contain a hole.
            section = solid.slice(100)
            self.assertAlmostEqual(section.area(), 160*120 - 140*100, delta=20)

    def test_sphere_offset(self):
        outer = trimesh.creation.icosphere(subdivisions=3, radius=80)
        inner, scene = make_inner(outer, wall=10, pitch=2, coarse_pitch=8, log=lambda _: None)
        check = validate_inner(inner, scene, 10, tolerance=1, samples=5000)
        self.assertTrue(check['passed'])
        self.assertAlmostEqual(inner.volume / (4/3*np.pi*70**3), 1., delta=.02)

    def test_scaling(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'box.stl'
            trimesh.creation.box([400, 100, 120]).export(path)
            mesh, transform = prepare_mesh(path, 8000, 'X')
            np.testing.assert_allclose(mesh.extents, [8000, 2000, 2400])
            self.assertEqual(transform['scale'], 20)

    def test_thin_body_rejected(self):
        with self.assertRaises(ValueError):
            make_inner(trimesh.creation.box([20, 20, 20]), wall=30, pitch=2,
                       coarse_pitch=8, log=lambda _: None)

    def test_open_mesh_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'open.stl'
            m = trimesh.creation.box()
            m.update_faces(np.arange(len(m.faces)-1))
            m.export(path)
            with self.assertRaises(ValueError):
                prepare_mesh(path)


if __name__ == '__main__':
    unittest.main()
