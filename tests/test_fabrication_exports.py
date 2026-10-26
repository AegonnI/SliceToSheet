import json
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import ezdxf
import manifold3d as md
import numpy as np
from shapely.geometry import Polygon

from marking import build_addresses,place_labels
from nesting import Packer,Item,nest_job,read_item,export_sheet
from pipeline import unpack
from sections import export_sections,section_contours,section_path,even_odd
from slicer import Part,Settings,from_solid,run


class FabricationExportTests(unittest.TestCase):
    def part(self,solid,pid='L001_Y001_X001_C01'):
        return Part(pid,(0,0,0),from_solid(solid))

    def test_three_numbers_are_unique_for_multiple_fragments(self):
        ids=['L001_Y006_X006_C01','L001_Y006_X006_C02','L001_Y006_X007_C01','L001_Y006_X007_C03']
        addresses=build_addresses(ids)
        self.assertEqual(addresses[ids[0]],'1 6 6')
        self.assertEqual(addresses[ids[2]],'1 6 7')
        self.assertEqual(len(set(addresses.values())),4)
        self.assertTrue(all(len(v.split())==3 and all(n.isdigit() for n in v.split()) for v in addresses.values()))
        self.assertEqual(addresses,build_addresses(reversed(ids)))

    def test_native_slice_does_not_use_trimesh_path_stitching(self):
        p=self.part(md.Manifold.cube([100,80,21])-md.Manifold.cube([60,40,30]).translate([20,20,-2]))
        with patch.object(p.mesh,'section',side_effect=AssertionError('Do not stitch paths')):
            _,contours=section_contours(p)
        self.assertEqual(len(contours),2)
        self.assertTrue(all(c['closed'] for c in contours))
        self.assertEqual(sorted(c['role'] for c in contours),['inner','outer'])

    def test_nested_material_hole_and_island_keep_even_odd_area(self):
        from shapely.geometry import box
        region=even_odd([box(0,0,20,20),box(5,5,15,15),box(8,8,12,12)])
        self.assertAlmostEqual(region.area,316)
        self.assertTrue(region.is_valid)

    def test_micro_slit_is_reported_and_real_hole_is_preserved(self):
        solid=md.Manifold.cube([100,100,21])-md.Manifold.cube([.002,20,30]).translate([50,30,-2])
        solid=solid-md.Manifold.cylinder(30,2.1,circular_segments=64).translate([20,20,-2])
        p=self.part(solid);original=p.mesh.volume
        _,contours=section_contours(p)
        self.assertEqual(len(contours),2)
        self.assertEqual(len(p.mesh.metadata['section_checks']['omitted_microcontours']),1)
        self.assertAlmostEqual(p.mesh.metadata['section_checks']['area_change_mm2'],.04,delta=.001)
        self.assertEqual(p.mesh.volume,original)

    def test_real_islands_are_not_deleted(self):
        solid=md.Manifold.cube([20,20,21])+md.Manifold.cube([20,20,21]).translate([40,0,0])
        solid=solid+md.Manifold.cube([60,20,4])
        p=self.part(solid)
        _,contours=section_contours(p)
        self.assertEqual(sum(c['role']=='outer' for c in contours),2)
        self.assertEqual(p.mesh.metadata['section_checks']['material_islands'],2)

    def test_rotated_labels_fit_real_material_and_dxf_width_matches_svg(self):
        p=self.part(md.Manifold.cube([4,90,21]))
        cfg=Settings(export_format='both',margin=2,svg_padding=0,nest_sheets=False)
        cfg._label_addresses={p.id:'123 12 34'}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);metrics,notes=export_sections(p,root,[210,297],cfg)
            self.assertFalse(notes)
            svg=ET.parse(section_path(root,p.id,'svg')).getroot()
            text=svg.find('.//{http://www.w3.org/2000/svg}text')
            self.assertIsNotNone(text.get('transform'))
            doc=ezdxf.readfile(section_path(root,p.id,'dxf'))
            label=next(e for e in doc.modelspace() if e.dxftype()=='TEXT' and e.dxf.layer=='LABELS')
            self.assertEqual(label.dxf.halign,5)
            self.assertAlmostEqual((label.dxf.align_point-label.dxf.insert).magnitude,float(text.get('textLength')),places=5)
            self.assertFalse(doc.audit().has_errors)

    def test_joint_label_placement_can_shrink_own_number_for_links(self):
        links=[dict(direction='UP',address='2 1 1',connection='K1',x=6,y=1),
               dict(direction='DOWN',address='1 1 2',connection='K2',x=6,y=11)]
        material=md.CrossSection.square([12,12])
        labels,notes=place_labels(material,'L001_Y001_X001_C01',[210,297],Settings(),np.zeros(2),'1 1 1',links)
        self.assertFalse(notes)
        texts=[ET.fromstring(x) for x in labels if ET.fromstring(x).tag=='text']
        self.assertEqual({x.text for x in texts},{'1 1 1','UP 2 1 1','DOWN 1 1 2'})
        for text in texts:
            self.assertGreaterEqual(float(text.get('font-size')),1.6)

    def test_true_contours_pack_diagonals_in_one_sheet(self):
        triangle=Polygon([(0,0),(100,0),(0,100)])
        p=Packer([110,110],2,7)
        for i in range(2):p.place(Item(str(i),triangle,None,0,''))
        self.assertEqual(len(p.boards),1)
        checks=p.verify()
        self.assertGreaterEqual(checks['minimum_gap_mm'],7)
        self.assertGreaterEqual(checks['minimum_edge_mm'],2-1e-6)
        self.assertEqual({p['rotation'] for p in p.placements},{0,180})

    def test_long_diagonal_parts_can_be_aligned_to_sheet(self):
        from shapely import affinity
        bar=affinity.rotate(Polygon([(0,0),(100,0),(100,8),(0,8)]),45,origin=(0,0))
        packer=Packer([110,110],2,7)
        for i in range(8):packer.place(Item(str(i),bar,None,0,''))
        self.assertLessEqual(len(packer.boards),2)
        self.assertTrue(any(abs(p['rotation']%90-45)<1e-5 for p in packer.placements))
        self.assertGreaterEqual(packer.verify()['minimum_gap_mm'],7)

    def test_nested_svg_dxf_fit_and_readable_rotation_agree(self):
        from tools.audit_exports import parse_svg,check_dxf
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=self.part(md.Manifold.cube([90,8,21]).rotate([0,0,30]))
            cfg=Settings(export_format='both',svg_padding=0,nest_sheets=False)
            export_sections(p,root/'sections_a4',[210,297],cfg)
            item=read_item(root/'sections_a4',p.id,'svg')
            export_sheet(1,[dict(id=p.id,sheet=1,rotation=180.,tx=100.,ty=100.)],{p.id:item},root,[210,297],cfg)
            data=parse_svg(root/'Sheets'/'SVG'/'Sheet_00001.svg')
            result=check_dxf(root/'Sheets'/'DXF'/'Sheet_00001.dxf',data)
            self.assertEqual(result['issues'],[])
            self.assertFalse(data['scaled']);self.assertFalse(data['reflected'])
            self.assertTrue(all(-90-1e-6<=t['angle']<=90+1e-6 for t in data['texts']))

    def test_old_export_can_be_recreated_without_slicing_or_drilling(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.stl'
            from_solid(md.Manifold.cube([80,60,42])).export(source)
            cfg=Settings(export_format='both',export_parts_stl=False,export_assembly_stl=False,nest_sheets=False)
            report=run(source,root/'result',cfg,log=lambda _:None)
            database=root/'result'/'progress.sqlite'
            con=sqlite3.connect(database)
            original=con.execute('SELECT id,mesh FROM parts ORDER BY id').fetchall()
            for pid,info in con.execute('SELECT id,info FROM parts').fetchall():
                data=json.loads(info);data.pop('export_revision',None)
                con.execute('UPDATE parts SET info=? WHERE id=?',(json.dumps(data),pid))
            con.commit();con.close()
            path=root/'result'/'assembly.json';data=json.loads(path.read_text(encoding='utf-8'));data.pop('export_revision',None);path.write_text(json.dumps(data),encoding='utf-8')
            with patch('slicer.iter_cut',side_effect=AssertionError('Do not reslice')),patch('slicer.place_connectors',side_effect=AssertionError('Do not drill')):
                fixed=run(source,root/'result',resume=True,log=lambda _:None)
            con=sqlite3.connect(database)
            self.assertEqual(original,con.execute('SELECT id,mesh FROM parts ORDER BY id').fetchall());con.close()
            self.assertEqual(fixed['volume_after_holes'],report['volume_after_holes'])
            self.assertTrue((root/'result'/'print_addresses.csv').is_file())

    def test_separate_nesting_export_keeps_single_part_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.stl'
            from_solid(md.Manifold.cube([420,40,21])).export(source)
            report=run(source,root/'result',Settings(export_format='both',margin=2,svg_padding=0,
                export_parts_stl=False,export_assembly_stl=False),log=lambda _:None)
            own=list((root/'result'/'sections_a4'/'SVG').rglob('*.svg'))
            nested=list((root/'result'/'Sheets'/'SVG').glob('*.svg'))
            self.assertEqual(len(own),len(report['parts']))
            self.assertEqual(len(nested),report['sheet_nesting']['sheets'])
            layout=json.loads((root/'result'/'Sheets'/'layout.json').read_text(encoding='utf-8'))
            self.assertEqual(len(layout['placements']),len(report['parts']))
            self.assertLess(len(nested),len(own))


if __name__=='__main__':unittest.main()
