"""Interactive desktop regression tests; run with the application's Qt runtime.

Set TEMP/TMP to a writable directory when running in the restricted sandbox.
The screenshot test writes the real Qt/VTK window to build/qa for inspection.
"""
import tempfile
import time
import unittest
import json
import sqlite3
from dataclasses import asdict
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import trimesh

try:
    from PySide6 import QtCore, QtGui, QtTest, QtWidgets
    import pyvista
    import pyvistaqt
except ImportError:
    QtCore = None


@unittest.skipIf(QtCore is None, 'Requires the PySide6 / PyVista application runtime')
class DesktopV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from desktop_ui import application
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
        QtCore.QSettings.setPath(QtCore.QSettings.Format.IniFormat,
                                 QtCore.QSettings.Scope.UserScope, str(cls.root/'preferences'))
        cls.application = application()
        cls.source = cls.root/'box.stl'
        cls.second_source = cls.root/'second.stl'
        trimesh.creation.box([100,60,40]).export(cls.source)
        trimesh.creation.box([80,40,20]).export(cls.second_source)

    @classmethod
    def tearDownClass(cls):
        cls.application.processEvents()
        cls.temp.cleanup()

    def setUp(self):
        from desktop_ui import App
        self.window = App()
        self.window.show_error = Mock()
        self.window.resize(1440,940)
        self.window.show()
        self.application.processEvents()

    def tearDown(self):
        self.wait_until(lambda: not self.window.busy)
        self.window.preview_timer.stop()
        self.window.close()
        self.application.processEvents()

    def wait_until(self, predicate, timeout=15):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            self.application.processEvents()
            if predicate():
                return
            QtTest.QTest.qWait(10)
        self.fail('Timed out waiting for Qt worker/event processing')

    def load(self, source=None):
        self.window.set_source(source or self.source)
        self.wait_until(lambda: not self.window.busy)
        self.assertIsNotNone(self.window.model_data)
        self.window.show_error.assert_not_called()

    def dimensions(self):
        return [self.window.dimensions[axis].value() for axis in 'XYZ']

    def test_linked_xyz_and_axis_independent_physical_scale(self):
        self.load()
        self.window.dimensions['X'].setValue(200)
        np.testing.assert_allclose(self.dimensions(), [200,120,80])
        self.window.dimensions['Y'].setValue(180)
        np.testing.assert_allclose(self.dimensions(), [300,180,120])
        for axis in 'XYZ':
            self.window.axis.setCurrentText(axis)
            cfg = self.window.cfg()
            self.assertEqual(cfg.scale, 3)
            self.assertEqual(cfg.target_length, 0)
            np.testing.assert_allclose(self.dimensions(), [300,180,120])

    def test_preview_is_gated_then_axis_changes_redraw_without_reload(self):
        self.load()
        self.window.draw = Mock(wraps=self.window.draw)
        self.window.axis.setCurrentText('Y')
        QtTest.QTest.qWait(220)
        self.assertFalse(self.window.preview_enabled)
        self.window.draw.assert_not_called()
        with patch('desktop_ui.prepare_preview') as prepare:
            self.window.preview_button.click()
            self.assertTrue(self.window.preview_enabled)
            self.assertEqual(self.window.draw.call_count, 1)
            first = self.window.draw.call_args.args[0][0]
            self.assertAlmostEqual(first.extents[2], 60)
            self.window.axis.setCurrentText('Z')
            self.wait_until(lambda: self.window.draw.call_count >= 2)
            updated = self.window.draw.call_args.args[0][0]
            self.assertAlmostEqual(updated.extents[2], 40)
            self.window.preview_button.click()
            self.assertEqual(self.window.draw.call_count, 3)
            prepare.assert_not_called()

    def test_source_switch_uses_new_geometry_and_discards_old_result(self):
        self.load()
        self.window.preview_button.click()
        self.window.result = ({'parts': []}, self.root)
        self.window.layer_button.setEnabled(True)
        self.window.dimensions['X'].setValue(200)
        self.load(self.second_source)
        self.assertIsNone(self.window.result)
        self.assertFalse(self.window.layer_button.isEnabled())
        self.assertEqual(self.window.model_data.source_key[0], str(self.second_source.resolve()))
        np.testing.assert_allclose(self.dimensions(), [80,40,20])
        self.assertTrue(self.window.preview_enabled)
        self.assertIn('Исходная модель', self.window.view_title.text())

    def test_preview_plan_uses_source_dimensions_even_if_proxy_bounds_shrink(self):
        from slicer import slicing_plan
        self.load()
        self.window.axis.setCurrentText('Z')
        self.window.model_data.mesh.apply_scale(.9)
        self.window.fields['thickness'].setValue(19)
        with patch('desktop_ui.slicing_plan', wraps=slicing_plan) as plan:
            self.window.preview_button.click()
            np.testing.assert_allclose(plan.call_args.args[0].extents, [100,60,40])
        self.assertIn('слоёв: 3', self.window.model_details.text())

    def test_settings_loaded_before_source_preserve_their_scale(self):
        from slicer import Settings
        self.window.apply_settings(Settings(scale=2, axis='Y'))
        self.load()
        self.assertEqual(self.window.model_scale, 2)
        self.assertEqual(self.window.axis.currentText(), 'Y')
        np.testing.assert_allclose(self.dimensions(), [200,120,80])

    def test_latest_settings_replace_pending_legacy_target_length(self):
        from slicer import Settings
        self.window.apply_settings(Settings(target_length=200, axis='X'))
        self.window.apply_settings(Settings(scale=3, axis='Y'))
        self.load()
        self.assertEqual(self.window.model_scale, 3)
        self.assertEqual(self.window.axis.currentText(), 'Y')
        np.testing.assert_allclose(self.dimensions(), [300,180,120])

    def test_settings_loaded_after_source_update_linked_dimensions(self):
        from slicer import Settings
        self.load()
        self.window.apply_settings(Settings(scale=2.5, axis='Z', export_format='both',
                                            export_parts_stl=False, export_assembly_stl=True))
        np.testing.assert_allclose(self.dimensions(), [250,150,100])
        cfg = self.window.cfg()
        self.assertEqual(cfg.export_format, 'both')
        self.assertFalse(cfg.export_parts_stl)
        self.assertTrue(cfg.export_assembly_stl)

    def test_resuming_different_source_clears_cache_and_reloads_after_job(self):
        from slicer import Settings
        self.load()
        self.window.preview_button.click()
        self.window.draw = Mock(wraps=self.window.draw)
        saved = Settings(scale=2,axis='Y')
        folder = self.root/'resume-second-source'
        folder.mkdir(exist_ok=True)
        with closing(sqlite3.connect(str(folder/'progress.sqlite'))) as db:
            with db:
                db.execute('CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT)')
                db.executemany('INSERT INTO meta VALUES (?,?)',
                               [('settings',json.dumps(asdict(saved))),
                                ('source',json.dumps(str(self.second_source)))])
        def launched(source, output, cfg, resume=False):
            self.window.busy = True
            self.window.current_output = output
        with patch.object(QtWidgets.QFileDialog,'getExistingDirectory',return_value=str(folder)), \
             patch.object(self.window,'launch_job',side_effect=launched) as launch:
            self.window.resume_job()
            launch.assert_called_once_with(str(self.second_source),folder,saved,resume=True)
            self.assertIsNone(self.window.model_data)
            self.assertTrue(all(not field.isEnabled() for field in self.window.dimensions.values()))
            self.assertTrue(self.window._reload_after_job)
            self.assertEqual(self.window._pending_settings,saved)
            # A preview request made during calculation must never draw the old
            # source. Once the job ends, the new source is read asynchronously.
            self.window.preview()
            self.window.draw.assert_not_called()
            self.window.task_idle()
            self.wait_until(lambda: not self.window.busy)
        self.window.show_error.assert_not_called()
        self.assertEqual(self.window.model_data.source_key[0],str(self.second_source.resolve()))
        self.assertFalse(self.window._reload_after_job)
        self.assertTrue(all(field.isEnabled() for field in self.window.dimensions.values()))
        np.testing.assert_allclose(self.dimensions(),[160,80,40])
        self.assertEqual(self.window.model_scale,2)
        self.assertEqual(self.window.axis.currentText(),'Y')
        self.assertGreater(self.window.draw.call_count,0)
        self.assertAlmostEqual(self.window.draw.call_args.args[0][0].extents[2],80)

    def test_unfocused_number_wheel_scrolls_sidebar_without_editing_value(self):
        for section in self.window.sections.values():
            section.toggle.setChecked(True)
        self.application.processEvents()
        scroll = self.window.scroll.verticalScrollBar()
        self.assertGreater(scroll.maximum(), 0)
        scroll.setValue(0)
        for field, value in ((self.window.fields['thickness'], lambda: self.window.fields['thickness'].value()),
                             (self.window.axis, lambda: self.window.axis.currentIndex())):
            with self.subTest(widget=type(field).__name__):
                scroll.setValue(0)
                self.window.preview_button.setFocus()
                self.application.processEvents()
                self.assertFalse(field.hasFocus())
                before = value()
                # Send through the real QWindow so Qt performs hit testing and
                # normal ancestor propagation, matching a physical mouse wheel.
                center = field.mapTo(self.window, field.rect().center())
                QtTest.QTest.wheelEvent(self.window.windowHandle(), center, QtCore.QPoint(0,-120))
                self.application.processEvents()
                self.assertEqual(value(), before)
                self.assertGreater(scroll.value(), 0)

    def test_optional_sections_collapse_and_theme_can_switch_both_ways(self):
        for name in ('shell','stringers','advanced','simplify'):
            self.assertTrue(self.window.sections[name].body.isHidden())
            self.window.sections[name].toggle.click()
            self.assertFalse(self.window.sections[name].body.isHidden())
        self.window.apply_theme(True)
        self.assertTrue(self.window.dark_theme)
        self.assertIn('#171b23', self.window.styleSheet())
        self.window.theme_button.click()
        self.assertFalse(self.window.dark_theme)
        self.assertIn('#edf1f5', self.window.styleSheet())
        self.window.theme_button.click()
        self.assertTrue(self.window.dark_theme)

    def test_stringer_settings_toggle_and_roundtrip(self):
        from slicer import Settings
        self.assertFalse(self.window.fields['stringer_depth'].isEnabled())
        cfg = Settings(stringers=True, stringer_spacing=250, stringer_depth=40,
                       stringer_height=21, stringer_up='-X', frame_layers=1)
        self.window.apply_settings(cfg)
        actual = self.window.cfg()
        for key in ('stringers', 'stringer_spacing', 'stringer_depth', 'stringer_height', 'stringer_up', 'frame_layers'):
            self.assertEqual(getattr(actual, key), getattr(cfg, key))
        self.assertTrue(self.window.fields['stringer_depth'].isEnabled())
        self.window.stringers.setChecked(False)
        self.assertFalse(self.window.fields['stringer_depth'].isEnabled())

    def test_real_window_screenshots_include_vtk_model_and_slicing_plane(self):
        self.load()
        self.window.preview_button.click()
        self.application.processEvents()
        self.assertIn('slicing_base', self.window.plotter.actors)
        self.assertIn('slicing_direction', self.window.plotter.actors)
        self.window.sections['stringers'].toggle.setChecked(True)
        self.window.stringers.setChecked(True)
        destination = Path(__file__).resolve().parents[1]/'build'/'qa'
        destination.mkdir(parents=True, exist_ok=True)
        for dark, name in ((True,'desktop-dark.png'), (False,'desktop-light.png')):
            self.window.apply_theme(dark)
            self.application.processEvents()
            pixmap = self.window.grab()
            self.assertFalse(pixmap.isNull())
            self.assertTrue(pixmap.save(str(destination/name)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
