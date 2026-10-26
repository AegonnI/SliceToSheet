import json
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh
from slicer import Settings, Part, place_connectors, run
from pipeline import Paused


class AssemblyUpdateTests(unittest.TestCase):
    def test_two_different_neighbors_below_and_above(self):
        def part(pid, cell, lower, upper):
            mesh=trimesh.creation.box(np.subtract(upper,lower))
            mesh.apply_translation((np.array(upper)+lower)/2)
            return Part(pid,cell,mesh)
        parts=[part('L001_Y001_X001_C01',(0,0,0),[0,0,0],[210,200,20]),
               part('L001_Y001_X002_C01',(1,0,0),[210,0,0],[420,200,20]),
               part('L002_Y001_X002_C01',(1,0,1),[105,0,20],[315,200,40]),
               part('L003_Y001_X001_C01',(0,0,2),[0,0,40],[210,200,60]),
               part('L003_Y001_X002_C01',(1,0,2),[210,0,40],[420,200,60])]
        records,_,_=place_connectors(parts,np.array([210,400,20]),Settings(staggered=True),lambda _:None)
        middle=parts[2].id
        down=[r for r in records if r['part_b']==middle]
        up=[r for r in records if r['part_a']==middle]
        self.assertEqual(len(down),2);self.assertEqual(len(up),2)
        self.assertEqual(len({r['part_a'] for r in down}),2)
        self.assertEqual(len({r['part_b'] for r in up}),2)
        self.assertEqual(len({(r['x'],r['y']) for r in down+up}),4)
        self.assertTrue(all(p.mesh.is_volume for p in parts))

    def test_limits_labels_and_side_switch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'box.stl'
            trimesh.creation.box([420,297,60]).export(source)
            for enabled in (False,True):
                out=root/str(enabled)
                r=run(source,out,Settings(staggered=True,margin=0,side_connections=enabled),log=lambda _:None)
                axes={c['axis'] for c in r['connections']}
                self.assertEqual(bool(axes-{'Z'}),enabled)
                counts=Counter((c[key],key) for c in r['connections'] if c['axis']=='Z' for key in ('part_a','part_b'))
                self.assertTrue(all(n<=2 for n in counts.values()))
                for p in r['parts']:
                    from sections import section_path
                    svg=section_path(out/'sections_a4',p['id'],'svg').read_text(encoding='utf-8')
                    self.assertIn(p['marking'],svg)
                    for link in p['links']:
                        from marking import link_text
                        text=link_text(link)
                        self.assertTrue(text in svg or any(text in note for note in p['marking_warnings']))

    def test_checkpoint_batches_and_soft_pause(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'box.stl';trimesh.creation.box([80,80,240]).export(source)
            import pipeline
            with patch('pipeline.pack',wraps=pipeline.pack) as pack:
                run(source,root/'base',Settings(),log=lambda _:None)
                # 12 raw parts, 11 at first connector checkpoint, then 2 at final checkpoint.
                self.assertEqual(pack.call_count,25)
            for stage in ('Нарезка','Отверстия'):
                stop=threading.Event();out=root/stage
                def progress(s,n,t):
                    if s==stage and n==2: stop.set()
                with self.assertRaises(Paused): run(source,out,Settings(),log=lambda _:None,stop=stop,progress=progress)
                db=sqlite3.connect(out/'progress.sqlite')
                try:
                    key='cut_next' if stage=='Нарезка' else 'connect_next'
                    self.assertEqual(json.loads(db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()[0]),2)
                finally: db.close()
                actual=run(source,out,resume=True,log=lambda _:None)
                baseline=json.loads((root/'base'/'assembly.json').read_text(encoding='utf-8'))
                self.assertEqual(actual['connections'],baseline['connections'])

    def test_crash_recovers_from_tenth_layer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'box.stl';trimesh.creation.box([80,80,240]).export(source)
            baseline=run(source,root/'base',Settings(),log=lambda _:None)
            for stage in ('Нарезка','Отверстия'):
                out=root/stage
                code="import sys,os;from slicer import run;run(sys.argv[1],sys.argv[2],log=lambda _:None,progress=lambda s,n,t:os._exit(77) if s==sys.argv[3] and n==11 else None)"
                result=subprocess.run([sys.executable,'-B','-c',code,str(source),str(out),stage],capture_output=True)
                self.assertEqual(result.returncode,77,result.stderr)
                db=sqlite3.connect(out/'progress.sqlite')
                try:
                    key='cut_next' if stage=='Нарезка' else 'connect_next'
                    self.assertEqual(json.loads(db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()[0]),10)
                finally: db.close()
                actual=run(source,out,resume=True,log=lambda _:None)
                self.assertEqual(actual['connections'],baseline['connections'])

    def test_legacy_options_and_validation(self):
        old=Settings.from_saved({'unique_keys':True,'connectors':3})
        old.validate();self.assertFalse(old.four_holes);self.assertTrue(old.side_connections)
        self.assertFalse(Settings().side_connections)
        for cfg in (Settings(unique_keys=True,connectors=3),Settings(checkpoint_layers=0),Settings(checkpoint_layers=1.5)):
            with self.assertRaises(ValueError): cfg.validate()


if __name__=='__main__': unittest.main(verbosity=2)
