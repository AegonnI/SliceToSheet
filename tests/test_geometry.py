import tempfile
import unittest
from pathlib import Path
import numpy as np
import trimesh
from slicer import Settings, load_model, cut, run, slicing_plan

class GeometryTests(unittest.TestCase):
    def test_native_topology_survives_coincident_vertices(self):
        import manifold3d
        from slicer import from_solid, as_solid
        # Two solids share an edge geometrically, but not topologically.
        a = manifold3d.Manifold.cube([2, 2, 2])
        solid = a + a.translate([2, 2, 0])
        mesh = from_solid(solid)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_volume)
        self.assertAlmostEqual(mesh.volume, 16)
        self.assertEqual(as_solid(mesh).status(), manifold3d.Error.NoError)
        self.assertEqual(len(mesh.split()), 2)

    def test_boat_scale_direction_and_large_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'boat.stl'
            trimesh.creation.box([400,115,130]).export(source)
            cfg=Settings(axis='X',reverse=True,target_length=8000,margin=0,scale=7)
            mesh,transform=load_model(source,cfg)
            self.assertTrue(np.allclose(mesh.extents,[2600,2300,8000]))
            nose=transform @ [200,0,0,1]
            stern=transform @ [-200,0,0,1]
            self.assertAlmostEqual(nose[2],0)
            self.assertAlmostEqual(stern[2],8000)
            steps,page,counts=slicing_plan(mesh,cfg)
            self.assertEqual(counts[2],400)
            self.assertGreater(np.prod(counts),5000)
            pieces,_,_=cut(mesh,cfg,lambda _:None,layer_indices=[0,399])
            self.assertEqual({p.cell[2] for p in pieces},{0,399})
            self.assertTrue(all(np.all(p.mesh.extents<=steps+1e-4) for p in pieces))
            self.assertAlmostEqual(sum(p.mesh.volume for p in pieces),2600*2300*40,places=1)

    def process(self, mesh, cfg):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name);source=root/'input.stl';mesh.export(source)
        report=run(source,root/'output',cfg,log=lambda _:None)
        return root/'output',report

    def test_a4_and_all_axes_connections(self):
        mesh=trimesh.creation.box([240,320,40])
        out,r=self.process(mesh,Settings(margin=0,diameter=10,connectors=1,pitch=25,side_connections=True,four_holes=False))
        self.assertEqual(len(r['parts']),8)
        self.assertEqual(set(c['axis'] for c in r['connections']),set('XYZ'))
        self.assertEqual(len(r['connected_groups']),1)
        for p in r['parts']:
            m=trimesh.load(out/'stl'/(p['id']+'.stl'),force='mesh')
            self.assertTrue(m.is_volume)
            self.assertTrue(np.all(m.extents <= np.array([210,297,20])+1e-4))
        self.assertLess(r['volume_after_holes'],r['volume_before_holes'])
        for c in r['connections']:
            self.assertAlmostEqual(c['hole_diameter'],10.2)
            # At both sides of the mating face a smaller coaxial cylinder must be empty.
            from slicer import cylinder, boolean
            center=np.array([c['x'],c['y'],c['z']]);axis='XYZ'.index(c['axis'])
            for key,sign in [('part_a',-1),('part_b',1)]:
                part=next(p for p in r['parts'] if p['id']==c[key])
                m=trimesh.load(out/'stl'/(part['id']+'.stl'),force='mesh')
                m.apply_translation(part['assembly_offset'])
                probe_center=center.copy();probe_center[axis]+=sign*5
                probe=cylinder(probe_center,axis,4.8,8)
                empty=boolean('intersection',[m,probe])
                self.assertTrue(empty.is_empty or abs(empty.volume)<1e-3)

    def test_short_last_layer_is_reported(self):
        out,r=self.process(trimesh.creation.box([80,80,25]),Settings())
        self.assertEqual(len(r['parts']),2)
        self.assertEqual(len(r['connections']),0)
        self.assertTrue(r['warnings'])

    def test_disconnected_shells_are_separate(self):
        a=trimesh.creation.box([15,15,15]);b=a.copy();b.apply_translation([40,0,0])
        out,r=self.process(trimesh.util.concatenate([a,b]),Settings())
        self.assertEqual(len(r['parts']),2)
        self.assertEqual(len(r['connected_groups']),2)

    def test_nonwatertight_rejected(self):
        m=trimesh.creation.box();m.update_faces(np.arange(len(m.faces)-1))
        with self.assertRaisesRegex(ValueError,'негерметичен'):self.process(m,Settings())

    def test_curved_volume_preserved_and_rotation(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'sphere.stl'
            sphere=trimesh.creation.icosphere(subdivisions=2,radius=32);sphere.export(source)
            c=Settings(axis='X',scale=2)
            m,t=load_model(source,c)
            pieces,steps,page=cut(m,c,lambda _:None)
            self.assertAlmostEqual(sum(p.mesh.volume for p in pieces)/m.volume,1,places=5)
            self.assertTrue(all(p.mesh.is_volume for p in pieces))
            self.assertTrue(np.allclose(m.extents,[128,128,128]))

if __name__=='__main__':unittest.main(verbosity=2)
