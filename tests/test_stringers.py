import tempfile
import threading
import unittest
from dataclasses import asdict
from pathlib import Path

import manifold3d as md
import numpy as np
import trimesh

from pipeline import Paused, load_config, save_config
from slicer import Settings, as_solid, from_solid, run
from stringers import ArcPath, StringerBuilder, side_paths


def shell(size=(220, 250, 80), wall=10):
    return from_solid(md.Manifold.cube(size)-md.Manifold.cube(np.asarray(size)-2*wall).translate([wall]*3))


class StringerTests(unittest.TestCase):
    def cfg(self, **values):
        defaults = dict(stringers=True, stringer_spacing=150, stringer_depth=40,
                        stringer_height=20, hull_wall=10, thickness=20,
                        export_parts_stl=False, export_assembly_stl=False)
        defaults.update(values)
        return Settings(**defaults)

    def test_keel_and_arc_length_on_both_sides(self):
        paths = side_paths([[10, 10], [210, 10], [210, 240], [10, 240]])
        self.assertEqual(len(paths), 2)
        for points in paths:
            np.testing.assert_allclose(points[0], [110, 10])
            self.assertEqual(points[-1, 1], 240)
            path = ArcPath(points, 2)
            self.assertAlmostEqual(path.length, 330)
            self.assertAlmostEqual(path.coordinate(path.interval(150, 150))[0], 150)

    def test_dimensions_both_sides_and_unchanged_outer_hull(self):
        mesh = shell()
        cfg = self.cfg()
        builder = StringerBuilder(mesh, cfg, log=lambda _: None)
        base = as_solid(mesh).trim_by_plane([0, 0, 1], 20).trim_by_plane([0, 0, -1], -40)
        result = builder.augment(base, 1)
        section = result.slice(30)
        # Two stringers per side (150 and 300 mm from the keel), 40 x 20 each.
        # Marching cubes bevels the eight inner corners by at most a grid cell.
        self.assertAlmostEqual(section.area()-base.slice(30).area(), 4*40*20, delta=20)
        self.assertLess((base-result).volume(), 1e-4)
        self.assertLess((result-md.Manifold.cube([220, 250, 80])).volume(), 1e-4)
        self.assertEqual(len(result.decompose()), 1)
        # 10 mm original wall + 40 mm rib: material ends at x=50, not x=40.
        self.assertGreater((result ^ md.Manifold.cube([1, 4, 4]).translate([48, 58, 28])).volume(), 15.9)
        self.assertLess((result ^ md.Manifold.cube([1, 4, 4]).translate([51, 58, 28])).volume(), .01)

    def test_adjacent_layers_have_identical_interface(self):
        mesh = shell()
        builder = StringerBuilder(mesh, self.cfg(), log=lambda _: None)
        solid = as_solid(mesh)
        sections = []
        for k in (1, 2):
            slab = solid.trim_by_plane([0, 0, 1], k*20).trim_by_plane([0, 0, -1], -(k+1)*20)
            result = builder.augment(slab, k)
            sections.append(result.slice(40+(-.001 if k == 1 else .001)))
        self.assertLess((sections[0]-sections[1]).area()+(sections[1]-sections[0]).area(), .05)

    def test_curved_cavity_depth_and_volume(self):
        outer = md.Manifold.cylinder(80, 100, circular_segments=96)
        cavity = md.Manifold.cylinder(60, 90, circular_segments=96).translate([0, 0, 10])
        mesh = from_solid(outer-cavity)
        mesh.apply_translation([100, 100, 0])
        cfg = self.cfg(stringer_spacing=80, stringer_depth=12, stringer_height=8)
        builder = StringerBuilder(mesh, cfg, log=lambda _: None)
        base = as_solid(mesh).trim_by_plane([0, 0, 1], 20).trim_by_plane([0, 0, -1], -40)
        result = builder.augment(base, 1)
        self.assertGreater(result.volume(), base.volume())
        self.assertLess((base-result).volume(), .001)
        self.assertLess((result-as_solid(mesh)-cavity.translate([100, 100, 0])).volume(), .001)
        self.assertEqual(len(result.decompose()), 1)
        # Three on each side, with finite width and the requested inward depth.
        self.assertAlmostEqual(result.slice(30).area()-base.slice(30).area(), 6*(8*12*(1-12/180)), delta=25)

    def test_combined_frames_volume_resume_and_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'shell.stl'
            shell((120, 130, 100)).export(source)
            cfg = self.cfg(stringer_spacing=65, stringer_depth=10, stringer_height=8,
                           frames=True, frame_interval=3, frame_coefficient=1.5,
                           checkpoint_layers=2, export_format='both', export_assembly_stl=True)
            baseline = run(source, root/'baseline', cfg, log=lambda _: None)
            self.assertGreater(baseline['stringers']['added_volume_mm3'], 0)
            self.assertGreater(baseline['frames']['added_volume_mm3'], 0)
            self.assertAlmostEqual(baseline['volume_before_holes'], shell((120, 130, 100)).volume+
                baseline['stringers']['added_volume_mm3']+baseline['frames']['added_volume_mm3'], places=2)
            stop = threading.Event()
            def progress(stage, number, total):
                if stage == 'Нарезка' and number == 3:
                    stop.set()
            with self.assertRaises(Paused):
                run(source, root/'paused', cfg, log=lambda _: None, stop=stop, progress=progress)
            resumed = run(source, root/'paused', resume=True, log=lambda _: None)
            for key in ('stringers', 'frames', 'metrics', 'volume_before_holes', 'volume_after_holes'):
                self.assertEqual(resumed[key], baseline[key])
            self.assertTrue(list((root/'paused'/'sections_a4').glob('*.svg')))
            self.assertTrue(list((root/'paused'/'sections_a4').glob('*.dxf')))
            assembled = trimesh.load(root/'paused'/'assembly.stl', force='mesh', process=False)
            self.assertAlmostEqual(assembled.volume, resumed['volume_after_holes'], delta=.1)

    def test_shell_creation_and_different_source_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'solid.stl'
            trimesh.creation.box([60, 70, 80]).export(source)
            report = run(source, root/'result', self.cfg(axis='X', scale=1.5, make_shell=True,
                         shell_pitch=2, stringer_spacing=45, stringer_depth=5, stringer_height=6), log=lambda _: None)
            self.assertTrue(report['shell']['closed_cavity'])
            self.assertGreater(report['stringers']['added_volume_mm3'], 0)
            self.assertEqual(report['size'][2], 90)

    def test_solid_requires_shell_and_cancel_is_observed(self):
        with self.assertRaisesRegex(ValueError, 'полость не найдена'):
            StringerBuilder(trimesh.creation.box([100]*3), self.cfg(), log=lambda _: None)
        def cancel():
            raise Paused()
        with self.assertRaises(Paused):
            StringerBuilder(shell(), self.cfg(), cancel=cancel, log=lambda _: None)

    def test_sloping_sides_and_other_up_direction(self):
        outer = md.Manifold.cylinder(100, 80, 110, circular_segments=48)
        inner = md.Manifold.cylinder(80, 72, 96, circular_segments=48).translate([0, 0, 10])
        mesh = from_solid(outer-inner)
        mesh.apply_translation([110, 110, 0])
        builder = StringerBuilder(mesh, self.cfg(stringer_spacing=80, stringer_depth=12,
                                 stringer_height=8, stringer_up='X'), log=lambda _: None)
        original = as_solid(mesh)
        for k in (1, 2):
            slab = original.trim_by_plane([0, 0, 1], k*20).trim_by_plane([0, 0, -1], -(k+1)*20)
            result = builder.augment(slab, k)
            self.assertEqual(result.status(), md.Error.NoError)
            self.assertEqual(len(result.decompose()), 1)
            self.assertGreater(result.volume(), slab.volume())
            self.assertLess((slab-result).volume(), .001)
            self.assertLess((result-(original+as_solid(from_solid(inner.translate([110, 110, 0]))))).volume(), .001)

    def test_settings_validation_roundtrip_and_legacy_defaults(self):
        for changes in ({'stringer_depth': 0}, {'stringer_spacing': float('nan')},
                        {'stringer_height': 250}, {'stringer_up': 'Z'}, {'stringers': 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.cfg(**changes).validate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'settings.json'
            cfg = self.cfg(stringer_up='X')
            save_config(path, cfg)
            self.assertEqual(load_config(path), cfg)
        legacy = {k: v for k, v in asdict(Settings()).items() if not k.startswith('stringer')}
        self.assertFalse(Settings.from_saved(legacy).stringers)

    def test_excessive_spacing_reports_no_ribs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'shell.stl'
            shell((60, 60, 40)).export(source)
            result = run(source, root/'output', self.cfg(stringer_spacing=1000), log=lambda _: None)
            self.assertEqual(result['stringers']['added_volume_mm3'], 0)
            self.assertTrue(any('Стрингеры' in line for line in result['warnings']))


if __name__ == '__main__':
    unittest.main()
