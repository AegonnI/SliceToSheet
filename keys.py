"""Asymmetric three-dowel codes, checked across complete mating triplets."""
import hashlib
from itertools import combinations, product
import numpy as np

class KeyRegistry:
    def __init__(self,tolerance):
        self.distance_tolerance=2*float(tolerance)
        self.buckets={}
        self.count=0

    def signature(self,points):
        p=np.asarray(points,dtype=float)
        distances=np.sort([np.linalg.norm(p[a]-p[b]) for a,b in ((0,1),(0,2),(1,2))])
        t=self.distance_tolerance
        # Scalene and visibly non-collinear, even allowing placement tolerance.
        height=np.linalg.norm(np.cross(p[1]-p[0],p[2]-p[0]))/max(distances[-1],1e-12)
        if height<=2*t or np.min(np.diff(distances))<=2*t:return None
        return distances

    def available(self,signature):
        if signature is None:return False
        s=np.asarray(signature);t=self.distance_tolerance
        cell=np.floor(s/t).astype(int)
        for delta in product((-1,0,1),repeat=3):
            key=tuple(cell+delta)
            for old in self.buckets.get(key,[]):
                if np.max(np.abs(s-old))<=t:return False
        return True

    def add(self,signature):
        s=np.asarray(signature,dtype=float)
        if not self.available(s):raise ValueError('Повторяющийся рисунок ключа в сохранённом расчёте.')
        key=tuple(np.floor(s/self.distance_tolerance).astype(int))
        self.buckets.setdefault(key,[]).append(s);self.count+=1


def place_keyed(parts,steps,cfg,log,owner_layer,occupied,cancel,registry):
    from slicer import neighbors,as_solid,from_solid,NativeMesh,probe_template,fits_native
    cfg.validate()
    registry=registry if registry is not None else KeyRegistry(cfg.key_tolerance)
    occupied=[] if occupied is None else occupied
    by_cell={}
    for p in parts:by_cell.setdefault(p.cell,[]).append(p)
    records=[];warnings=[]
    radius=(cfg.diameter+cfg.clearance)/2;safe_radius=radius+cfg.wall
    occupied_lo=np.asarray([a for a,b in occupied]).reshape(-1,3)
    occupied_hi=np.asarray([b for a,b in occupied]).reshape(-1,3)
    for a in parts:
        if owner_layer is not None and a.cell[2]!=owner_layer:continue
        for axis in range(3):
            for b,face in neighbors(a,axis,by_cell,steps,cfg):
                cancel()
                uv=[d for d in range(3) if d!=axis]
                lo=np.maximum(a.mesh.bounds[0],b.mesh.bounds[0]);hi=np.minimum(a.mesh.bounds[1],b.mesh.bounds[1])
                if np.any(hi[uv]-lo[uv]<=1e-5):continue
                key_id=a.id+'__'+b.id+'__'+'XYZ'[axis]
                seed=int.from_bytes(hashlib.sha256(key_id.encode()).digest()[:8],'little')
                rng=np.random.default_rng(seed)
                bore_half=cfg.length/2+(0.2 if axis!=2 else 0)
                safe_half=bore_half+(cfg.wall if axis!=2 else 0)
                low=lo[uv]+safe_radius;high=hi[uv]-safe_radius
                valid=[];envelopes=[];accepted=None
                if np.all(high>low):
                    # Stable per-interface random positions prevent identical grid codes.
                    for xy in rng.uniform(low,high,size=(192,2)):
                        cancel()
                        c=np.zeros(3);c[axis]=face;c[uv]=xy
                        bl=c-safe_radius;bh=c+safe_radius;bl[axis]=face-safe_half;bh[axis]=face+safe_half
                        if np.any(np.all(bh>occupied_lo,axis=1)&np.all(occupied_hi>bl,axis=1)):continue
                        ok=True
                        for piece,sign in ((a,-1),(b,1)):
                            center=c.copy();center[axis]+=sign*safe_half/2
                            extent=np.full(3,safe_radius);extent[axis]=safe_half/2
                            if np.any(center-extent<piece.mesh.bounds[0]-1e-4) or np.any(center+extent>piece.mesh.bounds[1]+1e-4):
                                ok=False;break
                            if not fits_native(piece.mesh,center,axis,safe_radius,safe_half):ok=False;break
                        if not ok:continue
                        valid.append(c);envelopes.append((bl,bh))
                        # Test triples involving the newest point; up to 18 verified candidates.
                        for i,j in combinations(range(len(valid)-1),2):
                            cancel()
                            indices=(i,j,len(valid)-1)
                            if any(np.all(envelopes[x][1]>envelopes[y][0]) and np.all(envelopes[y][1]>envelopes[x][0])
                                   for x,y in combinations(indices,2)):continue
                            points=[valid[x] for x in indices];signature=registry.signature(points)
                            if not registry.available(signature):continue
                            sa=as_solid(a.mesh);sb=as_solid(b.mesh)
                            for point in points:
                                bore=probe_template(axis,radius,2*bore_half).translate(point)
                                sa=sa-bore;sb=sb-bore
                            if any(s.is_empty() or s.volume()<=0 or len(s.decompose())!=1 for s in (sa,sb)):continue
                            accepted=(indices,signature,sa,sb);break
                        if accepted is not None or len(valid)>=18:break
                if accepted is None:
                    warnings.append('%s ↔ %s (%s): КЛЮЧ НЕ СОЗДАН — поиск не нашёл безопасный уникальный треугольник; отверстия этого стыка не вырезаны.'%(a.id,b.id,'XYZ'[axis]))
                    continue
                indices,signature,sa,sb=accepted
                registry.add(signature)
                a.mesh=NativeMesh(sa);b.mesh=NativeMesh(sb)
                for index in indices:
                    c=valid[index];bl,bh=envelopes[index];occupied.append((bl,bh))
                    occupied_lo=np.vstack((occupied_lo,bl));occupied_hi=np.vstack((occupied_hi,bh))
                    records.append(dict(id='K%04d'%(len(records)+1),part_a=a.id,part_b=b.id,axis='XYZ'[axis],
                        x=float(c[0]),y=float(c[1]),z=float(c[2]),hole_diameter=2*radius,dowel_diameter=cfg.diameter,
                        dowel_length=cfg.length,depth_each=bore_half,type='through_or_flush' if axis==2 else 'blind',
                        key_id=key_id,key_signature=signature.tolist()))
                log('%s ↔ %s: уникальный ключ, 3 шпильки'%(a.id,b.id))
    for p in parts:
        if isinstance(p.mesh,NativeMesh):
            solid=p.mesh.solid;p.mesh=from_solid(solid);p.mesh._cache['slicer_manifold']=solid
    # The streaming pipeline computes assembly-wide connectivity after all layers.
    if owner_layer is None:
        adjacency={p.id:set() for p in parts}
        for r in records:
            adjacency[r['part_a']].add(r['part_b']);adjacency[r['part_b']].add(r['part_a'])
        remaining=set(adjacency);groups=[]
        while remaining:
            todo=[remaining.pop()];group=[]
            while todo:
                p=todo.pop();group.append(p)
                for q in adjacency[p]:
                    if q in remaining:remaining.remove(q);todo.append(q)
            groups.append(sorted(group))
        if len(groups)>1:warnings.append('Сборка состоит из %d несвязанных групп деталей.'%len(groups))
    else:groups=[]
    return records,warnings,groups
