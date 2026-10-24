"""Geometry contracts for the simplified 1.1 interface and legacy projects."""
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import manifold3d as md
import numpy as np
import trimesh

from slicer import (Settings, _record_holes, cut, hole_surface_gap, iter_cut,
                    load_model, place_connectors, slicing_plan)


class GeometryV11Tests(unittest.TestCase):
    def source_box(self, folder, size):
        path = Path(folder)/'box.stl'
        trimesh.creation.box(size).export(path)
        return path

    def test_uniform_size_does_not_depend_on_slice_axis(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.source_box(folder, [80, 60, 40])
            for axis, along in zip('XYZ', [160, 120, 80]):
                mesh, transform = load_model(source, Settings(axis=axis, scale=2))
                np.testing.assert_allclose(sorted(mesh.extents), [80, 120, 160])
                self.assertAlmostEqual(mesh.extents[2], along)
                self.assertAlmostEqual(abs(np.linalg.det(transform[:3,:3])), 8)

    def test_legacy_target_length_and_reverse_remain_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.source_box(folder, [80, 60, 40])
            cfg = Settings.from_saved(dict(axis='X', target_length=160, scale=99, reverse=True))
            mesh, transform = load_model(source, cfg)
            self.assertFalse(cfg.auto_orientation)
            self.assertFalse(cfg.auto_dowel_length)
            self.assertEqual(cfg.min_hole_gap, 0)
            self.assertFalse(cfg.four_holes)
            self.assertTrue(cfg.side_connections)
            np.testing.assert_allclose(mesh.extents, [80, 120, 160])
            self.assertAlmostEqual((transform @ [40,0,0,1])[2], 0)

    def test_orientation_uses_fewer_estimated_sheets_and_keeps_geometry_axes(self):
        mesh = trimesh.creation.box([250, 150, 40])
        cfg = Settings(auto_orientation=True, margin=5)
        steps, page, counts = slicing_plan(mesh, cfg)
        self.assertEqual(page, (297., 210.))
        np.testing.assert_allclose(steps, [287, 200, 20])
        np.testing.assert_array_equal(counts, [1, 1, 2])
        self.assertFalse(cfg.landscape)  # selection does not overwrite a saved preference
        mesh = trimesh.creation.box([150, 250, 40])
        self.assertEqual(slicing_plan(mesh, cfg)[1], (210., 297.))

    def test_auto_short_last_layer_has_actual_pin_and_asymmetric_depths(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.source_box(folder, [80, 80, 25])
            cfg = Settings(auto_dowel_length=True, min_hole_gap=10)
            mesh, _ = load_model(source, cfg)
            parts, steps, _ = cut(mesh, cfg, lambda _: None)
            records, warnings, _ = place_connectors(parts, steps, cfg, lambda _: None)
            self.assertEqual(len(records), 2)
            self.assertFalse(warnings)
            for record in records:
                self.assertAlmostEqual(record['dowel_length'], 25)
                self.assertAlmostEqual(record['depth_a'], 20)
                self.assertAlmostEqual(record['depth_b'], 5)
                self.assertAlmostEqual(record['depth_each'], 5)
            self.assertTrue(all(p.mesh.is_volume for p in parts))

    def test_legacy_short_last_layer_still_reports_no_safe_connectors(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.source_box(folder, [80, 80, 25])
            cfg = Settings()
            mesh, _ = load_model(source, cfg)
            parts, steps, _ = cut(mesh, cfg, lambda _: None)
            records, _, _ = place_connectors(parts, steps, cfg, lambda _: None)
            self.assertEqual(records, [])

    def test_spacing_includes_both_faces_across_streaming_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.source_box(folder, [90, 80, 60])
            cfg = Settings(auto_dowel_length=True, min_hole_gap=25, wall=3)
            mesh, _ = load_model(source, cfg)
            parts, steps, _ = cut(mesh, cfg, lambda _: None)
            occupied, records = [], []
            for layer in (0, 1):
                batch = [p for p in parts if p.cell[2] in (layer, layer+1)]
                new, _, _ = place_connectors(batch, steps, cfg, lambda _: None,
                                             owner_layer=layer, occupied=occupied,
                                             hole_records=records)
                records.extend(new)
            self.assertEqual(len(records), 4)
            middle = next(p for p in parts if p.cell[2] == 1)
            holes = [hole for record in records for pid, hole in _record_holes(record)
                     if pid == middle.id]
            self.assertEqual(len(holes), 4)
            for i, first in enumerate(holes):
                for second in holes[i+1:]:
                    self.assertGreaterEqual(hole_surface_gap(first, second), 25-1e-5)
            for record in records:
                radius = record['hole_diameter']/2
                self.assertGreaterEqual(record['x']-radius, cfg.wall-1e-5)
                self.assertGreaterEqual(90-record['x']-radius, cfg.wall-1e-5)
                self.assertGreaterEqual(record['y']-radius, cfg.wall-1e-5)
                self.assertGreaterEqual(80-record['y']-radius, cfg.wall-1e-5)

    def test_finite_axial_segment_distance_checks_caps_and_crossed_axes(self):
        vertical = (np.array([0,0,0]), np.array([0,0,20]), 2)
        parallel = (np.array([10,0,0]), np.array([10,0,20]), 3)
        self.assertAlmostEqual(hole_surface_gap(vertical, parallel), 5)
        above = (np.array([0,0,30]), np.array([0,0,40]), 2)
        self.assertAlmostEqual(hole_surface_gap(vertical, above), 6)
        crossed = (np.array([-5,0,10]), np.array([5,0,10]), 2)
        self.assertAlmostEqual(hole_surface_gap(vertical, crossed), -4)

    def test_frame_augmentation_statistics_are_per_layer(self):
        cfg = Settings(frames=True, frame_interval=3)
        mesh = trimesh.creation.box([80,80,60])
        mesh.apply_translation(-mesh.bounds[0])
        class Builder:
            def __init__(self, *args):
                pass
            def augment(self, slab, index):
                if index == 2:
                    return slab + md.Manifold.cube([5,5,20]).translate([80,0,40])
                return slab
        module = types.SimpleNamespace(FrameBuilder=Builder)
        stats, added = {}, []
        with patch.dict('sys.modules', frames=module):
            for _, _ in iter_cut(mesh,cfg,lambda _: None,statistics=stats):
                added.append(stats['frame_added_volume'])
        np.testing.assert_allclose(added, [0, 0, 500])

    def test_native_simplification_in_final_millimetres_keeps_volume_topology(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'sphere.stl'
            trimesh.creation.icosphere(subdivisions=3, radius=20).export(source)
            cfg = Settings(scale=2, simplify_faces=200, simplify_tolerance=.2)
            mesh, _ = load_model(source, cfg)
            info = mesh.metadata['simplification']
            self.assertEqual(info['input_faces'], 1280)
            self.assertLessEqual(info['output_faces'], info['input_faces'])
            self.assertTrue(mesh.is_volume)
            self.assertEqual(info['tolerance_mm'], .2)
            self.assertIn(info['method'], ('manifold','quadric','unchanged'))
            if info['sampled_deviation_mm'] is not None:
                self.assertLessEqual(info['sampled_deviation_mm'], .2+1e-5)

    def test_coplanar_dense_box_reaches_face_target_without_changing_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'dense-box.stl'
            trimesh.creation.box([80,60,40]).subdivide().subdivide().export(source)
            mesh, _ = load_model(source, Settings(scale=2, simplify_faces=12))
            self.assertEqual(len(mesh.faces), 12)
            np.testing.assert_allclose(mesh.extents, [160,120,80])
            self.assertAlmostEqual(mesh.volume, 160*120*80)
            self.assertTrue(mesh.metadata['simplification']['target_reached'])

    def test_validation_of_new_settings(self):
        for settings in (dict(min_hole_gap=-1), dict(frame_interval=2),
                         dict(frame_coefficient=.5), dict(simplify_faces=3),
                         dict(simplify_tolerance=math.nan), dict(export_format='pdf')):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Settings(**settings).validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
