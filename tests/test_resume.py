import tempfile,threading,unittest,json,sqlite3
from pathlib import Path
from dataclasses import asdict
import trimesh,numpy as np
from unittest.mock import patch
from slicer import run,Settings
from pipeline import Paused,save_config,load_config

class ResumeTests(unittest.TestCase):
    def assert_same_report(self,actual,expected):
        # Parallel geometric reductions may differ at the last float bits.
        import copy
        a,b=copy.deepcopy(actual),copy.deepcopy(expected)
        for key in ('volume_before_holes','volume_after_holes'):
            self.assertAlmostEqual(a.pop(key),b.pop(key),places=6)
        for p,q in zip(a['parts'],b['parts']):
            self.assertAlmostEqual(p.pop('volume'),q.pop('volume'),places=6)
        self.assertEqual(a,b)

    def test_resume_each_stage_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input.stl';trimesh.creation.box([240,80,60]).export(source)
            cfg=Settings(connectors=1,pitch=25,margin=0)
            expected=run(source,root/'baseline',cfg,log=lambda _:None)
            for stage in ['Нарезка','Отверстия','Экспорт','Сборка STL']:
                with self.subTest(stage=stage):
                    stop=threading.Event();out=root/stage
                    def progress(s,n,total):
                        if s==stage and n>=1:stop.set()
                    with self.assertRaises(Paused):
                        run(source,out,cfg,log=lambda _:None,stop=stop,progress=progress)
                    self.assertTrue((out/'INCOMPLETE.txt').is_file())
                    actual=run(source,out,None,log=lambda _:None,resume=True)
                    self.assert_same_report(actual,expected)
                    self.assertFalse((out/'INCOMPLETE.txt').exists())
                    combined=trimesh.load(out/'assembly.stl',force='mesh')
                    self.assertAlmostEqual(combined.volume,actual['volume_after_holes'],places=1)
                    self.assertEqual(run(source,out,None,log=lambda _:None,resume=True),actual)

    def test_reject_changed_model_and_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input.stl';trimesh.creation.box([80,80,60]).export(source)
            original=source.read_bytes();stop=threading.Event();cfg=Settings()
            def progress(s,n,total):
                if s=='Нарезка':stop.set()
            with self.assertRaises(Paused):run(source,root/'job',cfg,log=lambda _:None,stop=stop,progress=progress)
            with self.assertRaisesRegex(ValueError,'Настройки'):
                run(source,root/'job',Settings(diameter=10),resume=True,log=lambda _:None)
            trimesh.creation.box([81,80,60]).export(source)
            with self.assertRaisesRegex(ValueError,'изменённый'):
                run(source,root/'job',resume=True,log=lambda _:None)
            source.write_bytes(original)
            self.assertTrue(run(source,root/'job',resume=True,log=lambda _:None)['parts'])

    def test_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'settings.json';cfg=Settings(axis='X',reverse=True,target_length=8000,margin=0)
            save_config(path,cfg);self.assertEqual(asdict(load_config(path)),asdict(cfg))
            data=json.loads(path.read_text(encoding='utf-8'));data['settings']['thickness']=-20
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):load_config(path)

    def test_recover_after_process_crash(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input.stl';trimesh.creation.box([80,80,60]).export(source)
            code="import os,sys; from slicer import run; run(sys.argv[1],sys.argv[2],log=lambda _:None,progress=lambda s,n,t:os._exit(77) if n==1 else None)"
            result=subprocess.run([sys.executable,'-B','-c',code,str(source),str(root/'job')],capture_output=True)
            self.assertEqual(result.returncode,77,result.stderr)
            report=run(source,root/'job',resume=True,log=lambda _:None)
            self.assertEqual(len(report['parts']),3)

    def test_pause_inside_connector_layer_rolls_back(self):
        import slicer
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input.stl';trimesh.creation.box([240,80,60]).export(source)
            cfg=Settings(connectors=1,pitch=25,margin=0)
            expected=run(source,root/'baseline',cfg,log=lambda _:None)
            stop=threading.Event();original=slicer.fits_native
            def interrupt(*args):
                result=original(*args);stop.set();return result
            with patch('slicer.fits_native',side_effect=interrupt):
                with self.assertRaises(Paused):run(source,root/'interrupted',cfg,log=lambda _:None,stop=stop)
            actual=run(source,root/'interrupted',resume=True,log=lambda _:None)
            self.assert_same_report(actual,expected)

if __name__=='__main__':unittest.main(verbosity=2)
