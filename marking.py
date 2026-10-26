"""Three-number print addresses and labels fitted to actual material."""
from collections import defaultdict
from html import escape
import math
import numpy as np
import manifold3d


def tokens(part_id):
    return tuple(int(token[1:]) for token in part_id.split('_'))


def build_addresses(part_ids):
    rows = defaultdict(lambda: defaultdict(list))
    for pid in sorted(set(part_ids)):
        layer, row, column, component = tokens(pid)
        rows[layer, row][column].append((component, pid))
    result = {}
    for (layer, row), columns in sorted(rows.items()):
        next_column = max(columns)+1
        extra = []
        for column, parts in sorted(columns.items()):
            parts.sort()
            result[parts[0][1]] = f'{layer} {row} {column}'
            extra.extend(pid for _, pid in parts[1:])
        for pid in extra:
            result[pid] = f'{layer} {row} {next_column}'
            next_column += 1
    if len(set(result.values())) != len(result):
        raise ValueError('Печатные номера деталей должны быть уникальными.')
    return result


def address(part_id, addresses=None):
    return (addresses or build_addresses([part_id]))[part_id]


def links_for(part_id, records, addresses=None):
    if addresses is None:
        addresses = build_addresses([part_id]+[r[k] for r in records for k in ('part_a','part_b')])
    result=[]
    for r in records:
        other = r['part_b'] if r['part_a'] == part_id else r['part_a']
        direction = ('UP' if r['part_a'] == part_id else 'DOWN') if r['axis'] == 'Z' else 'SIDE'
        result.append(dict(connection=r['id'], direction=direction, part=other,
            address=address(other,addresses), x=r['x'],y=r['y'],z=r['z']))
    return result


def link_text(link, unique_keys=False):
    text=link['direction']+' '+link['address']
    return text+' '+link['connection'] if unique_keys else text


def _candidates(section,low,high):
    from matplotlib.path import Path
    from scipy.ndimage import distance_transform_edt,maximum_filter
    pitch=max(.65,math.sqrt(float(np.prod(high-low))/40000))
    axes=[np.arange(low[i]+pitch/2,high[i],pitch) for i in range(2)]
    if any(len(a)==0 for a in axes):return np.empty((0,2)),np.empty(0),None
    points=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,2)
    mask=np.zeros(len(points),dtype=bool)
    paths=[];directions=defaultdict(float)
    for polygon in section.to_polygons():
        polygon=np.asarray(polygon)
        path=Path(np.vstack([polygon,polygon[0]]));paths.append(path)
        mask ^= path.contains_points(points)
        for delta in np.roll(polygon,-1,axis=0)-polygon:
            length=np.linalg.norm(delta)
            if length<2:continue
            angle=(math.degrees(math.atan2(delta[1],delta[0]))+90)%180-90
            directions[round(angle/5)*5]+=length
    shape=tuple(len(a) for a in axes)
    distances=distance_transform_edt(np.pad(mask.reshape(shape),1))[1:-1,1:-1]*pitch
    ridges=(distances>=maximum_filter(distances,size=3)-1e-7)&(distances>.6)
    sparse=np.zeros(shape,dtype=bool);sparse[::3,::3]=True
    selected=(ridges|sparse)&mask.reshape(shape)
    angles=[0.,90.]
    for angle,_ in sorted(directions.items(),key=lambda v:-v[1]):
        if all(abs((angle-a+90)%180-90)>7 for a in angles):angles.append(float(angle))
        if len(angles)>=5:break
    return points[selected.ravel()],distances[selected],(paths,angles)


def place_labels(section,part_id,page,cfg,origin,own_address,links):
    if section.is_empty():return [],[part_id+': среднее сечение пусто.']
    low,high=np.asarray(section.bounds()).reshape(2,2)
    centers,clearance,extra=_candidates(section,low,high)
    if not extra:return [],[part_id+': сечение слишком узкое для маркировки.']
    paths,angles=extra
    free=section.offset(-.25)
    labels=[];notes=[]
    sizes=(2.5,2.,1.6)
    def inside(points):
        mask=np.zeros(len(points),dtype=bool)
        for path in paths:mask ^= path.contains_points(points,radius=1e-8)
        return mask
    def place(text,near=None,connection=None):
        nonlocal free
        target=(low+high)/2 if near is None else np.asarray(near)
        order=np.argsort(np.linalg.norm(centers-target,axis=1)-clearance*2)
        for size in sizes:
            width,height=len(text)*size*.62,size*1.5
            eligible=order[clearance[order]>=height/2+.1][:700]
            variants=[]
            for angle in angles:
                rad=math.radians(angle)
                rotation=np.array([[math.cos(rad),-math.sin(rad)],[math.sin(rad),math.cos(rad)]])
                local=np.array([[-width/2,-height/2],[width/2,-height/2],[width/2,height/2],[-width/2,height/2]])
                offsets=local@rotation.T
                variants.append((angle,rotation,offsets))
            for index in eligible:
                for angle,rotation,offsets in variants:
                    center=centers[index];polygon=offsets+center
                    samples=np.vstack([polygon,(polygon+np.roll(polygon,-1,axis=0))/2,center])
                    if not inside(samples).all():continue
                    box=manifold3d.CrossSection([polygon])
                    if (box-free).area()>1e-5:continue
                    free=free-box.offset(.5)
                    baseline=center+rotation@[-width/2,-height/2+size*.3]
                    xy=baseline-origin+cfg.margin;x,y=xy[0],page[1]-xy[1]
                    transform='' if abs(angle)<1e-8 else ' transform="rotate(%.6f %.6f %.6f)"'%(-angle,x,y)
                    metadata='' if connection is None else ' data-connection="%s"'%escape(connection)
                    labels.append('<text x="%.6f" y="%.6f" font-family="monospace" font-size="%.2f" textLength="%.6f" lengthAdjust="spacingAndGlyphs"%s%s>%s</text>'%(x,y,size,width,transform,metadata,escape(text)))
                    if near is not None:
                        vector=center-np.asarray(near);distance=np.linalg.norm(vector)
                        if distance>1e-6:
                            end=np.asarray(near)+vector/distance*((cfg.diameter+cfg.clearance)/2+.4)
                            direction=rotation.T@(-vector/distance)
                            reach=min(width/2/max(abs(direction[0]),1e-12),height/2/max(abs(direction[1]),1e-12))
                            start=center-vector/distance*reach
                            a=start-origin+cfg.margin;b=end-origin+cfg.margin
                            labels.append('<path d="M %.6f %.6f L %.6f %.6f" fill="none" stroke="#777" stroke-width="0.12" data-connection="%s"/>'%(a[0],page[1]-a[1],b[0],page[1]-b[1],escape(connection or '')))
                    return True
        notes.append(part_id+': маркировка не помещается на сечении: '+text)
        return False
    requests=[(link_text(link,getattr(cfg,'unique_keys',False)),[link['x'],link['y']],link['connection']) for link in links]
    def attempt(maximum,own_first):
        nonlocal free,labels,notes,sizes
        free=section.offset(-.25);labels=[];notes=[]
        sizes=tuple(s for s in (2.5,2.,1.6) if s<=maximum)
        ordered=[(own_address,None,None),*requests] if own_first else [*sorted(requests,key=lambda r:-len(r[0])),(own_address,None,None)]
        own_ok=False
        for text,near,connection in ordered:
            ok=place(text,near,connection)
            if connection is None:own_ok=ok
        return labels,notes,(not own_ok,len(notes))
    best=attempt(2.5,True)
    if links and best[1]:
        for maximum,own_first in [(2.,True),(1.6,True),(2.,False),(1.6,False)]:
            candidate=attempt(maximum,own_first)
            if candidate[2]<best[2]:best=candidate
            if not best[1]:break
    return best[0],best[1]


def svg_labels(part,page,cfg,records,z):
    from sections import section_shape
    addresses=getattr(cfg,'_label_addresses',None)
    if addresses is None:
        addresses=build_addresses([part.id]+[r[k] for r in records for k in ('part_a','part_b')])
    _,section=section_shape(part)
    return place_labels(section,part.id,page,cfg,part.mesh.bounds[0,:2],address(part.id,addresses),links_for(part.id,records,addresses))
