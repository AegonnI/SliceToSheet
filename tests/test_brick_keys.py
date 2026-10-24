import unittest,tempfile,threading,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import trimesh
from slicer import Settings,load_model,cut,run,cylinder,boolean
from keys import KeyRegistry
from pipeline import Paused
from unittest.mock import patch

class BrickKeysTests(unittest.TestCase):
    def test_registry_rejects_rotations_reflections_and_near_duplicates(self):
        registry=KeyRegistry(.5)
        p=np.array([[0.,0,0],[40,0,0],[10,20,0]])
        s=registry.signature(p);self.assertIsNotNone(s);registry.add(s)
        self.assertFalse(registry.available(registry.signature(p[:,[1,0,2]]+100)))
        self.assertFalse(registry.available(s+.4))
        self.assertIsNone(registry.signature([[0,0,0],[20,0,0],[40,0,0]]))
        self.assertIsNone(registry.signature([[0,0,0],[40,0,0],[20,20,0]]))

    def test_staggered_volume_and_cross_seam_keys_with_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input.stl';trimesh.creation.box([420,297,40]).export(source)
            cfg=Settings(margin=0,staggered=True,unique_keys=True,connectors=3,four_holes=False,side_connections=True)
            mesh,_=load_model(source,cfg)
            parts,steps,_=cut(mesh,cfg,lambda _:None)
            self.assertAlmostEqual(sum(p.mesh.volume for p in parts),mesh.volume,places=2)
            self.assertTrue(all(np.all(p.mesh.extents<=steps+1e-4) for p in parts))
            upper=next(p for p in parts if p.cell==(1,0,1))
            self.assertTrue(np.allclose(upper.mesh.bounds[:,0],[105,315]))
            baseline=run(source,root/'baseline',cfg,log=lambda _:None)
            registry=KeyRegistry(cfg.key_tolerance);groups=defaultdict(list)
            for r in baseline['connections']:groups[r['key_id']].append(r)
            self.assertGreater(len(groups),0)
            for key,records in groups.items():
                self.assertEqual(len(records),3)
                points=np.array([[r['x'],r['y'],r['z']] for r in records])
                signature=registry.signature(points);self.assertTrue(registry.available(signature));registry.add(signature)
            lower_neighbors={r['part_a'] for r in baseline['connections'] if r['part_b']==upper.id and r['axis']=='Z'}
            self.assertEqual(len(lower_neighbors),2)
            for p in baseline['parts']:
                result=trimesh.load(root/'baseline'/'stl'/(p['id']+'.stl'),force='mesh')
                self.assertTrue(result.is_volume)
            stop=threading.Event()
            def progress(s,n,t):
                if s=='Отверстия' and n==1:stop.set()
            with self.assertRaises(Paused):run(source,root/'resume',cfg,log=lambda _:None,stop=stop,progress=progress)
            actual=run(source,root/'resume',resume=True,log=lambda _:None)
            self.assertEqual(actual['connections'],baseline['connections'])
            self.assertEqual(actual['key_validation'],baseline['key_validation'])
            # Stop after a key was accepted in RAM but before the layer transaction.
            stop.clear();original_add=KeyRegistry.add
            def interrupted_add(registry,signature):
                original_add(registry,signature);stop.set()
            with patch.object(KeyRegistry,'add',interrupted_add):
                with self.assertRaises(Paused):
                    run(source,root/'inside_key',cfg,log=lambda _:None,stop=stop)
            recovered=run(source,root/'inside_key',resume=True,log=lambda _:None)
            self.assertEqual(recovered['connections'],baseline['connections'])

    def test_no_partial_key_when_layer_too_short(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input.stl';trimesh.creation.box([80,80,25]).export(source)
            report=run(source,root/'out',Settings(unique_keys=True,connectors=3,four_holes=False),log=lambda _:None)
            self.assertEqual(report['connections'],[])
            self.assertTrue(any('КЛЮЧ НЕ СОЗДАН' in s for s in report['warnings']))
            self.assertAlmostEqual(report['volume_before_holes'],report['volume_after_holes'],places=5)

if __name__=='__main__':unittest.main(verbosity=2)
