import tempfile
import unittest
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np
import trimesh

from slicer import Settings, Part, export_svg, slicing_plan, load_model
from pipeline import save_config, load_config

NS = {'s': 'http://www.w3.org/2000/svg'}


class SvgPaddingTests(unittest.TestCase):
    def test_full_page_contours_and_labels_move_without_scaling(self):
        with tempfile.TemporaryDirectory() as temp:
            for page in ((210., 297.), (297., 210.)):
                mesh = trimesh.creation.box([*page, 20.])
                mesh.apply_translation([page[0]/2, page[1]/2, 10])
                part = Part('L001_Y001_X001_C01', (0, 0, 0), mesh)
                roots = []
                for padding in (0., 2., 5.):
                    cfg = Settings(margin=0, svg_padding=padding)
                    target = Path(temp)/'section.svg'
                    export_svg(part, target, page, cfg)
                    root = ET.parse(target).getroot()
                    roots.append(root)
                    document = (page[0]+2*padding, page[1]+2*padding)
                    self.assertEqual(root.attrib['width'], str(document[0])+'mm')
                    self.assertEqual(root.attrib['height'], str(document[1])+'mm')
                    self.assertEqual(list(map(float, root.attrib['viewBox'].split())), [0, 0, *document])
                    wrapper = root.find('s:g', NS)
                    self.assertEqual(wrapper.attrib, {'transform': 'translate(%s %s)' % (padding, padding)})
                    self.assertGreater(len(wrapper.findall('s:text', NS)), 1)
                    for line in wrapper.findall('s:g/s:polyline', NS):
                        xy = np.array([list(map(float, p.split(','))) for p in line.attrib['points'].split()])
                        self.assertTrue(np.allclose(xy[0], xy[-1]))
                        self.assertTrue(np.allclose(np.ptp(xy, axis=0), page))
                        # A 0.2 mm stroke is wholly inside the enlarged page.
                        if padding:
                            self.assertTrue(np.all(xy.min(axis=0)+padding-.1 > 0))
                            self.assertTrue(np.all(xy.max(axis=0)+padding+.1 < document))
                # All drawing elements are unchanged: only the enclosing translation differs.
                for root in roots[1:]:
                    original = roots[0].find('s:g', NS)
                    padded = root.find('s:g', NS)
                    self.assertEqual([ET.tostring(c) for c in original], [ET.tostring(c) for c in padded])

    def test_padding_does_not_change_model_or_slicing_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'boat.stl'
            trimesh.creation.box([400, 115, 130]).export(source)
            baseline = None
            for padding in (0., 2., 100.):
                cfg = Settings(axis='X', target_length=8000, margin=0, svg_padding=padding)
                mesh, transform = load_model(source, cfg)
                steps, page, counts = slicing_plan(mesh, cfg)
                values = (mesh.vertices, transform, steps, np.array(page), counts)
                if baseline is None:
                    baseline = values
                else:
                    for a, b in zip(baseline, values):
                        np.testing.assert_array_equal(a, b)

    def test_saved_settings_and_old_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'settings.json'
            cfg = Settings(svg_padding=3.5)
            save_config(path, cfg)
            self.assertEqual(asdict(load_config(path)), asdict(cfg))
            old = asdict(cfg)
            del old['svg_padding']
            self.assertEqual(Settings.from_saved(old).svg_padding, 0)

    def test_invalid_padding(self):
        for value in (-1, 101, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Settings(svg_padding=value).validate()


if __name__ == '__main__':
    unittest.main()
