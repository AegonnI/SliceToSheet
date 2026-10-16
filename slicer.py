"""Exact solid STL slicing and conservative dowel placement. All coordinates in mm."""
from dataclasses import dataclass, asdict
from pathlib import Path
from itertools import product
from functools import lru_cache
import csv
import json
import math
import numpy as np
import trimesh
import manifold3d

@dataclass
class Settings:
    thickness: float = 20.0
    diameter: float = 8.0
    clearance: float = 0.2  # diametral clearance
    length: float = 40.0
    wall: float = 2.0
    margin: float = 5.0
    svg_padding: float = 2.0  # extra document space in mm; never changes model or cutting grid
    scale: float = 1.0
    axis: str = 'Z'
    landscape: bool = False
    connectors: int = 2
    pitch: float = 12.0
    target_length: float = 0.0  # uniform fit along the selected source axis, mm
    reverse: bool = False  # start from the maximum of the source axis
    staggered: bool = False
    unique_keys: bool = False
    key_tolerance: float = 0.5
    four_holes: bool = True
    side_connections: bool = False
    checkpoint_layers: int = 10

    @classmethod
    def from_saved(cls, values):
        # Preserve the geometry algorithm of jobs made before these options existed.
        values = dict(values)
        values.setdefault('four_holes', False)
        values.setdefault('side_connections', True)
        values.setdefault('svg_padding', 0.0)  # keep existing jobs' SVG layout
        return cls(**values)

    def validate(self):
        if self.four_holes and self.unique_keys:
            raise ValueError('Режим четырёх отверстий несовместим с тройками ключей.')
        if type(self.checkpoint_layers) is not int or self.checkpoint_layers < 1:
            raise ValueError('Интервал сохранения должен быть целым числом >= 1.')
        numbers = [self.thickness, self.diameter, self.clearance, self.length,
                   self.wall, self.margin, self.scale, self.pitch, self.target_length, self.key_tolerance, self.svg_padding]
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError('Параметры должны быть конечными числами.')
        if min(self.thickness, self.diameter, self.length, self.scale, self.pitch) <= 0:
            raise ValueError('Толщина, диаметр, длина, масштаб и шаг должны быть > 0.')
        if min(self.clearance, self.wall, self.margin) < 0 or self.margin >= 90:
            raise ValueError('Зазор и стенка >= 0; поля от 0 до 90 мм (не включая 90).')
        if self.target_length < 0:
            raise ValueError('Длина вдоль оси должна быть >= 0.')
        if not 0 <= self.svg_padding <= 100:
            raise ValueError('Свободное пространство SVG должно быть от 0 до 100 мм.')
        if self.key_tolerance <= 0 or (self.unique_keys and self.key_tolerance < self.clearance):
            raise ValueError('Допуск ключа должен быть положительным и не меньше зазора отверстия.')
        if self.unique_keys and self.connectors != 3:
            raise ValueError('Для уникальных ключей нужны ровно 3 шпильки на стык.')
        if self.axis not in ('X', 'Y', 'Z') or self.connectors not in (1, 2, 3, 4):
            raise ValueError('Неверная ось или количество соединителей.')

@dataclass
class Part:
    id: str
    cell: tuple
    mesh: object


class NativeMesh:
    """Keep successive booleans native until the layer is ready for storage."""
    def __init__(self,solid):
        self.solid=solid
        self.bounds=np.asarray(solid.bounding_box()).reshape(2,3)


def boolean(operation, meshes):
    solids = [as_solid(m) for m in meshes]
    result = solids[0]
    for other in solids[1:]:
        if operation == 'intersection': result = result ^ other
        elif operation == 'difference': result = result - other
        elif operation == 'union': result = result + other
        else: raise ValueError('Unknown boolean operation: '+operation)
    return from_solid(result)


def as_solid(mesh):
    if isinstance(mesh,NativeMesh):return mesh.solid
    key = 'slicer_manifold'
    if key not in mesh._cache:
        mesh._cache[key] = manifold3d.Manifold(manifold3d.Mesh(
            vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
            tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
    return mesh._cache[key]


def from_solid(solid):
    data = solid.to_mesh()
    # Manifold already supplies indexed, valid topology. Coincident vertices
    # can belong to distinct edges; trimesh's default welding destroys those.
    return trimesh.Trimesh(vertices=data.vert_properties[:, :3], faces=data.tri_verts,
                          process=False)


def cylinder(center, axis, radius, length):
    direction = np.eye(3)[axis]
    transform = trimesh.geometry.align_vectors([0, 0, 1], direction)
    transform[:3, 3] = center
    return trimesh.creation.cylinder(radius=radius, height=length,
                                    sections=64, transform=transform)


def load_model(path, cfg):
    cfg.validate()
    is_step = Path(path).suffix.lower() in ('.step', '.stp')
    kind = 'STEP/STP' if is_step else 'STL'
    if is_step:
        from step_import import load_step
        mesh = load_step(path)
    else:
        mesh = trimesh.load(str(path), force='mesh', process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f'{kind} не содержит треугольников.')
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f'Некорректные координаты в {kind}.')
    if not mesh.is_watertight:
        raise ValueError(f'{kind} негерметичен: закройте отверстия в редакторе и повторите.')
    # A valid hollow solid has inward-oriented cavity components. Unconditionally
    # fixing each component independently would turn those cavities into solids.
    if not mesh.is_winding_consistent or not mesh.is_volume:
        trimesh.repair.fix_normals(mesh, multibody=True)
    if not mesh.is_volume:
        raise ValueError(f'{kind} не задаёт корректный замкнутый объём.')
    axis_index = 'XYZ'.index(cfg.axis)
    if min(mesh.extents) <= 1e-5:
        raise ValueError('Модель имеет нулевую толщину.')
    factor = cfg.target_length / mesh.extents[axis_index] if cfg.target_length else cfg.scale
    transform = trimesh.geometry.align_vectors(np.eye(3)[axis_index], [0, 0, -1 if cfg.reverse else 1])
    transform[:3, :3] *= factor
    mesh.apply_transform(transform)
    shift = np.eye(4)
    shift[:3, 3] = -mesh.bounds[0]
    mesh.apply_transform(shift)
    if min(mesh.extents) <= 1e-5:
        raise ValueError('Модель имеет нулевую толщину.')
    return mesh, shift @ transform


def slicing_plan(mesh, cfg):
    page = (297., 210.) if cfg.landscape else (210., 297.)
    steps = np.array([page[0]-2*cfg.margin, page[1]-2*cfg.margin, cfg.thickness])
    counts = np.maximum(1, np.ceil((mesh.extents-1e-5)/steps).astype(int))
    if np.prod(counts+(np.array([1,1,0]) if cfg.staggered else 0)) > 1000000:
        raise ValueError('Более миллиона ячеек. Проверьте размеры модели и толщину слоя.')
    return steps, page, counts


def grid_origin(layer,steps,cfg):
    return -np.asarray(steps[:2])/2 if cfg.staggered and layer%2 else np.zeros(2)


def neighbors(a,axis,by_cell,steps,cfg):
    if axis<2:
        if not cfg.side_connections:
            return
        cell=list(a.cell);cell[axis]+=1
        face=grid_origin(a.cell[2],steps,cfg)[axis]+cell[axis]*steps[axis]
        for b in by_cell.get(tuple(cell),[]):yield b,face
    else:
        layer=a.cell[2]+1;origin=grid_origin(layer,steps,cfg)
        lo=np.maximum(0,np.floor((a.mesh.bounds[0,:2]-origin+1e-5)/steps[:2]).astype(int))
        hi=np.ceil((a.mesh.bounds[1,:2]-origin-1e-5)/steps[:2]).astype(int)
        for i,j in product(range(lo[0],hi[0]),range(lo[1],hi[1])):
            for b in by_cell.get((i,j,layer),[]):yield b,layer*steps[2]


def iter_cut(mesh, cfg, log, layer_indices=None, cancel=lambda: None):
    steps, page, counts = slicing_plan(mesh, cfg)
    # Convert the dense scan once. First clip a transverse slab, then its A4 tiles.
    solid = manifold3d.Manifold(manifold3d.Mesh(
        vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
    if solid.status() != manifold3d.Error.NoError:
        raise ValueError('Не удалось преобразовать модель в объём для нарезки.')
    mesh._cache.clear()
    block = None
    block_number = -1
    for k in (range(counts[2]) if layer_indices is None else layer_indices):
        cancel()
        parts = []
        if not 0 <= k < counts[2]:
            raise ValueError('Номер слоя вне модели.')
        # Only every 16th layer touches the complete scan. Subsequent planes
        # work on a much smaller exact block, without simplifying geometry.
        if k//16 != block_number:
            block_number = k//16
            block = solid.trim_by_plane([0,0,1],block_number*16*steps[2]).trim_by_plane(
                [0,0,-1],-min((block_number+1)*16*steps[2],mesh.bounds[1,2]))
        slab = block.trim_by_plane([0, 0, 1], k*steps[2]).trim_by_plane(
            [0, 0, -1], -min((k+1)*steps[2], mesh.bounds[1, 2]))
        if slab.is_empty():
            yield k, parts
            continue
        bounds = np.asarray(slab.bounding_box()).reshape(2, 3)
        origin=grid_origin(k,steps,cfg)
        start = np.maximum(0, np.floor((bounds[0, :2]-origin)/steps[:2]).astype(int))
        stop = np.ceil((bounds[1, :2]-origin-1e-5)/steps[:2]).astype(int)
        for j, i in product(range(start[1], stop[1]), range(start[0], stop[0])):
            cancel()
            if i == start[0]:
                row = slab.trim_by_plane([0,1,0],origin[1]+j*steps[1]).trim_by_plane([0,-1,0],-origin[1]-(j+1)*steps[1])
            tile = row.trim_by_plane([1,0,0],origin[0]+i*steps[0]).trim_by_plane([-1,0,0],-origin[0]-(i+1)*steps[0])
            if tile.is_empty() or tile.volume() < 1e-5:
                continue
            clipped = from_solid(tile)
            for n, component in enumerate(clipped.split(only_watertight=False), 1):
                if component.volume > 1e-5:
                    if not component.is_volume:
                        raise ValueError('Ошибка геометрии после нарезки: '
                                         f'слой {k+1}, ячейка X{i+1} Y{j+1}, компонент {n}.')
                    parts.append(Part('L%03d_Y%03d_X%03d_C%02d' % (k+1,j+1,i+1,n),
                                      (i,j,k), component))
                    component._cache.clear()
        log('Слой %d/%d, деталей: %d' % (k+1, counts[2], len(parts)))
        yield k, parts


def cut(mesh, cfg, log, layer_indices=None):
    steps,page,_ = slicing_plan(mesh,cfg)
    parts = [p for _, batch in iter_cut(mesh,cfg,log,layer_indices) for p in batch]
    return parts, steps, page


def fits(mesh, probe):
    overlap = as_solid(mesh) ^ as_solid(probe)
    return not overlap.is_empty() and abs(overlap.volume()-probe.volume) <= max(1e-4, probe.volume*1e-5)


@lru_cache(maxsize=64)
def probe_template(axis,radius,length):
    solid=manifold3d.Manifold.cylinder(length,radius,circular_segments=64,center=True)
    return solid.rotate([0,90,0] if axis==0 else [-90,0,0] if axis==1 else [0,0,0])


def fits_native(mesh,center,axis,radius,length):
    template=probe_template(axis,radius,length)
    overlap=as_solid(mesh) ^ template.translate(center)
    volume=template.volume()
    return not overlap.is_empty() and abs(overlap.volume()-volume)<=max(1e-4,volume*1e-5)


def connector_pairs(parts, by_cell, steps, cfg, owner_layer, usage):
    """One hole per neighbor first; a second pass fills remaining capacity."""
    layers = sorted({p.cell[2] for p in parts if owner_layer is None or p.cell[2] == owner_layer})
    for layer in layers:
        vertical = []
        for a in sorted((p for p in parts if p.cell[2] == layer), key=lambda p: p.id):
            for axis in range(3):
                for b, face in neighbors(a, axis, by_cell, steps, cfg):
                    if axis == 2 and cfg.four_holes:
                        if np.all(np.minimum(a.mesh.bounds[1,:2], b.mesh.bounds[1,:2]) >
                                  np.maximum(a.mesh.bounds[0,:2], b.mesh.bounds[0,:2])):
                            vertical.append((a, axis, b, face))
                    else:
                        yield a, axis, b, face
        for _ in range(2):
            pending = list(vertical)
            while pending:
                # Give unconnected parts priority before filling the second slot.
                pending.sort(key=lambda pair: (
                    usage.get((pair[0].id, 'up'), 0) + usage.get((pair[2].id, 'down'), 0),
                    pair[0].id, pair[2].id))
                yield pending.pop(0)


def place_connectors(parts, steps, cfg, log, owner_layer=None, occupied=None, cancel=lambda: None, key_registry=None):
    if cfg.unique_keys:
        from keys import place_keyed
        return place_keyed(parts,steps,cfg,log,owner_layer,occupied,cancel,key_registry)
    by_cell = {}
    for p in parts:
        by_cell.setdefault(p.cell, []).append(p)
    records, warnings = [], []
    occupied = [] if occupied is None else occupied
    occupied_lo=np.array([a for a,b in occupied]).reshape(-1,3)
    occupied_hi=np.array([b for a,b in occupied]).reshape(-1,3)
    radius = (cfg.diameter+cfg.clearance)/2
    safe_radius = radius+cfg.wall
    usage = {}
    for a, axis, b, face in connector_pairs(parts, by_cell, steps, cfg, owner_layer, usage):
        cancel()
        target = cfg.connectors
        if cfg.four_holes and axis == 2:
            if usage.get((a.id, 'up'), 0) >= 2 or usage.get((b.id, 'down'), 0) >= 2:
                continue
            target = 1
        uv = [x for x in range(3) if x != axis]
        lo = np.maximum(a.mesh.bounds[0], b.mesh.bounds[0])
        hi = np.minimum(a.mesh.bounds[1], b.mesh.bounds[1])
        if any(hi[t]-lo[t] <= 1e-5 for t in uv):
            continue
        candidates = []
        spans = []
        for t in uv:
            width = hi[t]-lo[t]-2*safe_radius
            spans.append(np.linspace(lo[t]+safe_radius, hi[t]-safe_radius,
                         min(35, max(1, int(width/cfg.pitch)+1))) if width >= 0 else [])
        for u, v in product(*spans):
            c = np.zeros(3); c[axis] = face; c[uv] = [u, v]
            candidates.append(c)
        selected = []
        sorted_for = -1
        # Alternate interfaces naturally select different positions due to collision checks.
        while candidates and len(selected) < target:
            cancel()
            if sorted_for != len(selected):
                points=np.asarray(candidates)
                if selected:
                    scores=np.linalg.norm(points[:,None,:]-np.asarray(selected)[None,:,:],axis=2).min(axis=1)
                    order=np.argsort(scores,kind='stable')
                else:
                    scores=np.linalg.norm(points[:,uv]-(lo[uv]+hi[uv])/2,axis=1)
                    order=np.argsort(-scores,kind='stable')
                candidates=[candidates[index] for index in order]
                sorted_for = len(selected)
            c = candidates.pop()
            half = cfg.length/2
            # Side bores have 0.2 mm axial relief and a closed end with wall allowance.
            # Z bores can end flush at the exterior of a complete 20 mm layer.
            bore_half = half + (0.2 if axis != 2 else 0.0)
            safe_half = bore_half + (cfg.wall if axis != 2 else 0.0)
            bound_lo = c-np.full(3, safe_radius)
            bound_hi = c+np.full(3, safe_radius)
            bound_lo[axis] = c[axis]-safe_half
            bound_hi[axis] = c[axis]+safe_half
            if np.any(np.all(bound_hi > occupied_lo,axis=1) & np.all(occupied_hi > bound_lo,axis=1)):
                continue
            ok = True
            for piece, sign in ((a,-1), (b,1)):
                center = c.copy(); center[axis] += sign*safe_half/2
                extent = np.full(3,safe_radius);extent[axis]=safe_half/2
                if np.any(center-extent < piece.mesh.bounds[0]-1e-4) or np.any(center+extent > piece.mesh.bounds[1]+1e-4):
                    ok=False;break
                if not fits_native(piece.mesh,center,axis,safe_radius,safe_half):
                    ok = False; break
            if not ok:
                continue
            bore = probe_template(axis,radius,2*bore_half).translate(c)
            solid_a=as_solid(a.mesh)-bore;solid_b=as_solid(b.mesh)-bore
            if any(m.is_empty() or m.volume()<=0 or len(m.decompose())!=1 for m in (solid_a,solid_b)):
                continue
            a.mesh, b.mesh = NativeMesh(solid_a),NativeMesh(solid_b)
            selected.append(c)
            if cfg.four_holes and axis == 2:
                usage[(a.id, 'up')] = usage.get((a.id, 'up'), 0) + 1
                usage[(b.id, 'down')] = usage.get((b.id, 'down'), 0) + 1
            occupied.append((bound_lo, bound_hi))
            occupied_lo=np.vstack((occupied_lo,bound_lo))
            occupied_hi=np.vstack((occupied_hi,bound_hi))
            records.append(dict(id='K%04d' % (len(records)+1), part_a=a.id, part_b=b.id,
                                axis='XYZ'[axis], x=float(c[0]), y=float(c[1]), z=float(c[2]),
                                hole_diameter=2*radius, dowel_diameter=cfg.diameter,
                                dowel_length=cfg.length, depth_each=bore_half,
                                type='through_or_flush' if axis==2 else 'blind'))
        if len(selected) < target and not (cfg.four_holes and axis == 2):
            warnings.append('%s ↔ %s (%s): %d/%d шпилек; не найдено достаточно мест с запасом материала.'
                            % (a.id,b.id,'XYZ'[axis],len(selected),cfg.connectors))
        log('%s ↔ %s: %d соединений' % (a.id,b.id,len(selected)))
    for p in parts:
        if isinstance(p.mesh,NativeMesh):
            solid=p.mesh.solid;p.mesh=from_solid(solid);p.mesh._cache['slicer_manifold']=solid
    if owner_layer is not None:
        return records, warnings, []
    linked = {r[key] for r in records for key in ('part_a','part_b')}
    for p in parts:
        if p.id not in linked and len(parts)>1:
            warnings.append(p.id+': НЕТ СОЕДИНЕНИЙ. Требуется ручная доработка.')
    # Report disconnected assembly groups, even when every part has some connection.
    remaining = {p.id for p in parts}; groups=[]
    while remaining:
        reached={next(iter(remaining))}
        while True:
            old=len(reached)
            for r in records:
                if r['part_a'] in reached or r['part_b'] in reached:
                    reached.update((r['part_a'],r['part_b']))
            if len(reached)==old: break
        groups.append(sorted(reached)); remaining-=reached
    if len(groups)>1:
        warnings.append('Сборка состоит из %d несвязанных групп деталей.' % len(groups))
    return records, warnings, groups


def export_svg(part, destination, page, cfg, records=()):
    # A middle section is a dimensioned drilling/section reference, not a solid projection.
    origin = part.mesh.bounds[0].copy()
    z = float(part.mesh.bounds[:,2].mean())
    section = part.mesh.section(plane_origin=[0,0,z], plane_normal=[0,0,1])
    padding = cfg.svg_padding
    document = (page[0] + 2*padding, page[1] + 2*padding)
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="%smm" height="%smm" viewBox="0 0 %s %s">' % (*document,*document),
             '<rect width="100%" height="100%" fill="white"/>',
             '<g transform="translate(%s %s)">' % (padding,padding),
             '<g fill="none" stroke="black" stroke-width="0.2">']
    if section is not None:
        for contour in section.discrete:
            xy=contour[:,:2]-origin[:2]+cfg.margin
            points=' '.join('%.4f,%.4f' % (x,page[1]-y) for x,y in xy)
            lines.append('<polyline points="%s"/>' % points)
    from marking import svg_labels
    labels, notes = svg_labels(part, page, cfg, records, z)
    lines += ['</g>'] + labels + ['<text x="3" y="4" font-size="2.5">%s | MID SECTION z=%.3f mm | PRINT 100%%</text>' % (part.id,z), '</g>', '</svg>']
    destination.write_text('\n'.join(lines), encoding='utf-8')
    return notes


def run(source, output, cfg=None, log=print, resume=False, stop=None, progress=None):
    from pipeline import run_job
    return run_job(source,output,cfg,log,resume,stop,progress)


if __name__=='__main__':
    import argparse
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    parser=argparse.ArgumentParser(description='STL / STEP / STP → 20 mm layers / A4 / dowels')
    parser.add_argument('stl', metavar='model', help='Input STL, STEP or STP file');parser.add_argument('output')
    parser.add_argument('--diameter',type=float,choices=[8,10],default=8)
    parser.add_argument('--scale',type=float,default=1)
    parser.add_argument('--axis',choices=['X','Y','Z'],default='Z')
    parser.add_argument('--target-length',type=float,default=0,help='Length along source axis in mm, preserving proportions')
    parser.add_argument('--reverse',action='store_true',help='Start at maximum source coordinate')
    parser.add_argument('--margin',type=float,default=5)
    parser.add_argument('--svg-padding',type=float,default=2,help='Extra SVG document space on each side in mm; preserves model dimensions')
    parser.add_argument('--resume',action='store_true',help='Resume the output folder using its saved settings')
    parser.add_argument('--staggered',action='store_true',help='Alternate half-page offsets in adjacent layers')
    parser.add_argument('--unique-keys',action='store_true',help='Use checked asymmetric three-dowel codes')
    parser.add_argument('--legacy-holes',action='store_true',help='Use the old per-interface hole count')
    parser.add_argument('--side-connections',action='store_true',help='Enable lateral dowels within a layer')
    parser.add_argument('--checkpoint-layers',type=int,default=10)
    args=parser.parse_args()
    run(args.stl,args.output,None if args.resume else Settings(diameter=args.diameter,scale=args.scale,axis=args.axis,
        target_length=args.target_length,reverse=args.reverse,margin=args.margin,svg_padding=args.svg_padding,
        staggered=args.staggered,unique_keys=args.unique_keys,connectors=3 if args.unique_keys else 2,
        four_holes=not (args.legacy_holes or args.unique_keys),side_connections=args.side_connections,
        checkpoint_layers=args.checkpoint_layers),resume=args.resume)
