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
    auto_orientation: bool = False
    min_hole_gap: float = 0.0  # minimum material between two hole surfaces, mm
    auto_dowel_length: bool = False
    frames: bool = False
    hull_wall: float = 50.0
    frame_interval: int = 25
    frame_coefficient: float = 2.0
    make_shell: bool = False
    shell_pitch: float = 3.0
    simplify_faces: int = 0  # zero keeps the input mesh unchanged
    simplify_tolerance: float = 0.5  # millimetres after scaling
    export_format: str = 'svg'
    export_parts_stl: bool = True
    export_assembly_stl: bool = True

    @classmethod
    def from_saved(cls, values):
        # Preserve the geometry algorithm of jobs made before these options existed.
        values = dict(values)
        values.setdefault('four_holes', False)
        values.setdefault('side_connections', True)
        values.setdefault('svg_padding', 0.0)  # keep existing jobs' SVG layout
        values.setdefault('auto_orientation', False)
        values.setdefault('auto_dowel_length', False)
        values.setdefault('min_hole_gap', 0.0)
        values.setdefault('frames', False)
        return cls(**values)

    def validate(self):
        if self.four_holes and self.unique_keys:
            raise ValueError('Режим четырёх отверстий несовместим с тройками ключей.')
        if type(self.checkpoint_layers) is not int or self.checkpoint_layers < 1:
            raise ValueError('Интервал сохранения должен быть целым числом >= 1.')
        for name in ('auto_orientation', 'auto_dowel_length', 'frames', 'make_shell',
                     'export_parts_stl', 'export_assembly_stl'):
            if type(getattr(self, name)) is not bool:
                raise ValueError('Настройка %s должна быть включена или выключена.' % name)
        if type(self.frame_interval) is not int or self.frame_interval < 3:
            raise ValueError('Интервал шпангоутов должен быть целым числом >= 3.')
        if type(self.simplify_faces) is not int or self.simplify_faces < 0:
            raise ValueError('Число полигонов должно быть целым числом >= 0.')
        if self.simplify_faces and self.simplify_faces < 12:
            raise ValueError('Целевое число полигонов должно быть не меньше 12.')
        if self.export_format not in ('svg', 'dxf', 'both'):
            raise ValueError('Формат экспорта: SVG, DXF или оба.')
        numbers = [self.thickness, self.diameter, self.clearance, self.length,
                   self.wall, self.margin, self.scale, self.pitch, self.target_length,
                   self.key_tolerance, self.svg_padding, self.min_hole_gap, self.hull_wall,
                   self.frame_coefficient, self.shell_pitch, self.simplify_tolerance]
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError('Параметры должны быть конечными числами.')
        if min(self.thickness, self.diameter, self.length, self.scale, self.pitch) <= 0:
            raise ValueError('Толщина, диаметр, длина, масштаб и шаг должны быть > 0.')
        if self.min_hole_gap < 0:
            raise ValueError('Минимальная перемычка между отверстиями должна быть >= 0.')
        if min(self.hull_wall, self.shell_pitch, self.simplify_tolerance) <= 0:
            raise ValueError('Толщина корпуса, шаг оболочки и допуск упрощения должны быть > 0.')
        if self.frame_coefficient < 1:
            raise ValueError('Коэффициент ширины шпангоута должен быть >= 1.')
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


def _surface_samples(mesh, count=2048):
    """Deterministic area-weighted samples; never change the caller's RNG."""
    rng = np.random.default_rng(1100)
    area = mesh.area_faces
    indices = rng.choice(len(mesh.faces), count, p=area/area.sum())
    weights = rng.random((count, 2))
    flip = weights.sum(axis=1) > 1
    weights[flip] = 1-weights[flip]
    triangles = mesh.vertices[mesh.faces[indices]]
    return (triangles[:, 0] + weights[:, :1]*(triangles[:, 1]-triangles[:, 0]) +
            weights[:, 1:]*(triangles[:, 2]-triangles[:, 0]))


def _sampled_surface_deviation(original, candidate):
    """Bidirectional sampled distance, an estimate rather than a Hausdorff bound."""
    try:
        import rtree  # trimesh's exact nearest-triangle queries require this index
    except ImportError:
        return None
    worst = 0.0
    for source, target in ((original, candidate), (candidate, original)):
        points = _surface_samples(source)
        for start in range(0, len(points), 128):
            _, distances, _ = trimesh.proximity.closest_point(target, points[start:start+128])
            if not np.isfinite(distances).all():
                return math.inf
            worst = max(worst, float(np.max(distances)))
    return worst


def simplify_model(mesh, cfg):
    """Optional simplification in final millimetres, with conservative rejection.

    Native simplification bounds surface movement by the requested tolerance.
    A QEM candidate can improve the face count, but is accepted only after
    topology, volume and sampled bidirectional distance checks. The latter is
    explicitly a sampled measurement, not a guarantee for every surface point.
    """
    if not cfg.simplify_faces or len(mesh.faces) <= cfg.simplify_faces:
        return mesh
    original = as_solid(mesh)
    if original.status() != manifold3d.Error.NoError:
        raise ValueError('Нельзя упростить некорректный объём модели.')
    topology = (len(original.decompose()), original.genus())
    original_volume = float(original.volume())
    tolerance = cfg.simplify_tolerance
    info = dict(input_faces=len(mesh.faces), target_faces=cfg.simplify_faces,
                tolerance_mm=tolerance, output_faces=len(mesh.faces),
                method='unchanged', sampled_deviation_mm=None,
                sample_count_each_direction=2048,
                validation='topology, <=1% volume change; sampled distance is not a global bound')
    best = mesh

    def acceptable(candidate, method, needs_distance):
        nonlocal best
        if not isinstance(candidate, trimesh.Trimesh) or len(candidate.faces) >= len(best.faces):
            return False
        if not np.isfinite(candidate.vertices).all() or not candidate.is_volume:
            return False
        solid = as_solid(candidate)
        if solid.status() != manifold3d.Error.NoError or solid.is_empty():
            return False
        if (len(solid.decompose()), solid.genus()) != topology:
            return False
        relative_volume = abs(solid.volume()-original_volume)/original_volume
        if relative_volume > 0.01:
            return False
        deviation = _sampled_surface_deviation(mesh, candidate)
        if needs_distance and deviation is None:
            return False
        if deviation is not None and deviation > tolerance+1e-5:
            return False
        best = candidate
        info.update(method=method, output_faces=len(candidate.faces),
                    sampled_deviation_mm=deviation, volume_change_fraction=relative_volume)
        return True

    acceptable(from_solid(original.simplify(tolerance)), 'manifold', False)
    if len(best.faces) > cfg.simplify_faces:
        try:
            candidate = mesh.simplify_quadric_decimation(face_count=cfg.simplify_faces)
        except (ImportError, ModuleNotFoundError):
            info['qem_note'] = 'Optional fast-simplification dependency is unavailable.'
        except (ValueError, RuntimeError) as error:
            info['qem_note'] = 'QEM candidate could not be built: '+str(error)
        else:
            if not acceptable(candidate, 'quadric', True):
                info['qem_note'] = 'QEM candidate rejected by topology, volume or sampled-distance checks.'
    info['target_reached'] = len(best.faces) <= cfg.simplify_faces
    best.metadata['simplification'] = info
    return best


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
    mesh = simplify_model(mesh, cfg)
    return mesh, shift @ transform


def slicing_plan(mesh, cfg):
    page = (297., 210.) if cfg.landscape else (210., 297.)
    if cfg.auto_orientation:
        # Estimate rectangular tiles for all layers, including the half-page
        # offsets of odd layers. No model/label rotation is needed: only A4's
        # width and height change, so all exported text remains upright.
        layers = max(1, int(math.ceil((mesh.extents[2]-1e-5)/cfg.thickness)))
        def score(candidate):
            available = np.asarray(candidate)-2*cfg.margin
            plain = np.maximum(1, np.ceil((mesh.extents[:2]-1e-5)/available)).astype(int)
            shifted = np.maximum(1, np.ceil((mesh.extents[:2]-1e-5)/available+.5)).astype(int)
            even = (layers+1)//2 if cfg.staggered else layers
            odd = layers//2 if cfg.staggered else 0
            tiles = int(np.prod(plain))*even + int(np.prod(shifted))*odd
            # Fewer splits first, balanced tile dimensions as a stable tie-break;
            # preserve the manual preference if both estimates are equivalent.
            imbalance = abs(float(mesh.extents[0]/available[0]-mesh.extents[1]/available[1]))
            return tiles, imbalance, candidate != page
        page = min(((210., 297.), (297., 210.)), key=score)
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


def iter_cut(mesh, cfg, log, layer_indices=None, cancel=lambda: None, statistics=None, frame_log=None):
    steps, page, counts = slicing_plan(mesh, cfg)
    # Convert the dense scan once. First clip a transverse slab, then its A4 tiles.
    solid = manifold3d.Manifold(manifold3d.Mesh(
        vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
    if solid.status() != manifold3d.Error.NoError:
        raise ValueError('Не удалось преобразовать модель в объём для нарезки.')
    mesh._cache.clear()
    frame_builder = None
    if cfg.frames and counts[2] >= cfg.frame_interval:
        from frames import FrameBuilder
        frame_builder = FrameBuilder(mesh, cfg, cancel, frame_log or log)
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
        added_volume = 0.0
        if frame_builder is not None:
            before = slab.volume()
            slab = frame_builder.augment(slab, k)
            added_volume = slab.volume()-before
            if added_volume < -max(.01, abs(before)*1e-5):
                raise ValueError('Шпангоут удалил материал исходного корпуса.')
        if statistics is not None:
            statistics['frame_added_volume'] = max(0.0, added_volume)
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


def _hole_segment(center, axis, depth, sign):
    end = np.asarray(center, dtype=float).copy()
    end[axis] += sign*depth
    return np.minimum(center, end), np.maximum(center, end)


def hole_surface_gap(first, second):
    """Exact minimum distance between the axes of two axial bores, minus radii.

    Each item contains (segment_low, segment_high, radius). Axis-aligned line
    segments permit independent interval distances in the three coordinates.
    A negative value means the cylindrical envelopes can overlap. This is a
    conservative material gap at a closed axial end as well as along a wall.
    """
    low_a, high_a, radius_a = first
    low_b, high_b, radius_b = second
    separation = np.maximum(0, np.maximum(np.asarray(low_a)-high_b, np.asarray(low_b)-high_a))
    return float(np.linalg.norm(separation)-radius_a-radius_b)


def _record_holes(record):
    center = np.array([record['x'], record['y'], record['z']], dtype=float)
    axis = 'XYZ'.index(record['axis'])
    radius = record['hole_diameter']/2
    for key, depth_key, sign in (('part_a', 'depth_a', -1), ('part_b', 'depth_b', 1)):
        depth = record.get(depth_key, record['depth_each'])
        yield record[key], (*_hole_segment(center, axis, depth, sign), radius)


def place_connectors(parts, steps, cfg, log, owner_layer=None, occupied=None, cancel=lambda: None,
                     key_registry=None, hole_records=()):
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
    spread_holes = cfg.auto_dowel_length or cfg.min_hole_gap > 0
    part_holes = {p.id: [] for p in parts}
    if spread_holes:
        for record in hole_records:
            for pid, hole in _record_holes(record):
                if pid in part_holes:
                    part_holes[pid].append(hole)
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
        half = cfg.thickness if cfg.auto_dowel_length else cfg.length/2
        if cfg.auto_dowel_length and axis == 2:
            # A short final layer receives a through hole to its own exterior;
            # the other layer retains its complete thickness. The pin is the
            # sum of these two actual depths, rather than being rejected.
            depth_a = min(half, max(0.0, face-a.mesh.bounds[0, axis]))
            depth_b = min(half, max(0.0, b.mesh.bounds[1, axis]-face))
        else:
            depth_a = depth_b = half+(0.2 if axis != 2 else 0.0)
        if min(depth_a, depth_b) <= 1e-5:
            continue
        safe_depth_a = depth_a+(cfg.wall if axis != 2 else 0.0)
        safe_depth_b = depth_b+(cfg.wall if axis != 2 else 0.0)
        selected = []
        sorted_for = -1
        # Alternate interfaces naturally select different positions due to collision checks.
        while candidates and len(selected) < target:
            cancel()
            if sorted_for != len(selected):
                points=np.asarray(candidates)
                if spread_holes:
                    scores = []
                    existing = [(a.id, depth_a, -1), (b.id, depth_b, 1)]
                    for point in points:
                        gaps = [hole_surface_gap((*_hole_segment(point, axis, depth, sign), radius), previous)
                                for pid, depth, sign in existing for previous in part_holes[pid]]
                        # Seed at an extremity of the allowed area; subsequent
                        # holes maximize their worst gap to all holes of either
                        # piece, including holes facing the preceding layer.
                        scores.append(min(gaps) if gaps else
                                      float(np.linalg.norm(point[uv]-(lo[uv]+hi[uv])/2)))
                    order=np.argsort(scores,kind='stable')
                elif selected:
                    scores=np.linalg.norm(points[:,None,:]-np.asarray(selected)[None,:,:],axis=2).min(axis=1)
                    order=np.argsort(scores,kind='stable')
                else:
                    scores=np.linalg.norm(points[:,uv]-(lo[uv]+hi[uv])/2,axis=1)
                    order=np.argsort(-scores,kind='stable')
                candidates=[candidates[index] for index in order]
                sorted_for = len(selected)
            c = candidates.pop()
            if spread_holes:
                too_close = any(hole_surface_gap((*_hole_segment(c, axis, depth, sign), radius), previous)
                                < cfg.min_hole_gap-1e-5
                                for pid, depth, sign in ((a.id, depth_a, -1), (b.id, depth_b, 1))
                                for previous in part_holes[pid])
                if too_close:
                    continue
            bound_lo = c-np.full(3, safe_radius)
            bound_hi = c+np.full(3, safe_radius)
            bound_lo[axis] = c[axis]-safe_depth_a
            bound_hi[axis] = c[axis]+safe_depth_b
            if not spread_holes and np.any(np.all(bound_hi > occupied_lo,axis=1) & np.all(occupied_hi > bound_lo,axis=1)):
                continue
            ok = True
            for piece, sign, safe_depth in ((a,-1,safe_depth_a), (b,1,safe_depth_b)):
                center = c.copy(); center[axis] += sign*safe_depth/2
                extent = np.full(3,safe_radius);extent[axis]=safe_depth/2
                if np.any(center-extent < piece.mesh.bounds[0]-1e-4) or np.any(center+extent > piece.mesh.bounds[1]+1e-4):
                    ok=False;break
                if not fits_native(piece.mesh,center,axis,safe_radius,safe_depth):
                    ok = False; break
            if not ok:
                continue
            bore_center = c.copy();bore_center[axis] += (depth_b-depth_a)/2
            bore = probe_template(axis,radius,depth_a+depth_b).translate(bore_center)
            solid_a=as_solid(a.mesh)-bore;solid_b=as_solid(b.mesh)-bore
            if any(m.is_empty() or m.volume()<=0 or len(m.decompose())!=1 for m in (solid_a,solid_b)):
                continue
            a.mesh, b.mesh = NativeMesh(solid_a),NativeMesh(solid_b)
            selected.append(c)
            if spread_holes:
                part_holes[a.id].append((*_hole_segment(c,axis,depth_a,-1),radius))
                part_holes[b.id].append((*_hole_segment(c,axis,depth_b,1),radius))
            if cfg.four_holes and axis == 2:
                usage[(a.id, 'up')] = usage.get((a.id, 'up'), 0) + 1
                usage[(b.id, 'down')] = usage.get((b.id, 'down'), 0) + 1
            occupied.append((bound_lo, bound_hi))
            occupied_lo=np.vstack((occupied_lo,bound_lo))
            occupied_hi=np.vstack((occupied_hi,bound_hi))
            records.append(dict(id='K%04d' % (len(records)+1), part_a=a.id, part_b=b.id,
                                axis='XYZ'[axis], x=float(c[0]), y=float(c[1]), z=float(c[2]),
                                hole_diameter=2*radius, dowel_diameter=cfg.diameter,
                                dowel_length=(depth_a+depth_b if cfg.auto_dowel_length else cfg.length),
                                depth_each=min(depth_a,depth_b),
                                type='through_or_flush' if axis==2 else 'blind'))
            if cfg.auto_dowel_length:
                records[-1].update(depth_a=depth_a,depth_b=depth_b)
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
    parser.add_argument('--auto-orientation',action='store_true',help='Estimate the A4 orientation with fewer tiles')
    parser.add_argument('--auto-dowel-length',action='store_true',help='Use the actual thickness of the two adjoining layers')
    parser.add_argument('--min-hole-gap',type=float,default=0,help='Minimum material between hole surfaces, mm')
    parser.add_argument('--export-format',choices=['svg','dxf','both'],default='svg')
    parser.add_argument('--no-parts-stl',action='store_true',help='Omit the individual STL folder')
    parser.add_argument('--no-assembly-stl',action='store_true',help='Omit assembly.stl')
    parser.add_argument('--frames',action='store_true',help='Add two-layer frames inside a closed cavity')
    parser.add_argument('--hull-wall',type=float,default=50,help='Final hull wall thickness, mm')
    parser.add_argument('--frame-interval',type=int,default=25)
    parser.add_argument('--frame-coefficient',type=float,default=2)
    parser.add_argument('--make-shell',action='store_true',help='Create a cavity before slicing')
    parser.add_argument('--shell-pitch',type=float,default=3,help='Shell distance-field grid step, mm')
    parser.add_argument('--simplify-faces',type=int,default=0,help='Optional target triangle count; zero disables simplification')
    parser.add_argument('--simplify-tolerance',type=float,default=.5,help='Allowed simplification deviation in final mm')
    args=parser.parse_args()
    run(args.stl,args.output,None if args.resume else Settings(diameter=args.diameter,scale=args.scale,axis=args.axis,
        target_length=args.target_length,reverse=args.reverse,margin=args.margin,svg_padding=args.svg_padding,
        staggered=args.staggered,unique_keys=args.unique_keys,connectors=3 if args.unique_keys else 2,
        four_holes=not (args.legacy_holes or args.unique_keys),side_connections=args.side_connections,
        checkpoint_layers=args.checkpoint_layers,auto_orientation=args.auto_orientation,
        auto_dowel_length=args.auto_dowel_length,min_hole_gap=args.min_hole_gap,
        export_format=args.export_format,export_parts_stl=not args.no_parts_stl,
        export_assembly_stl=not args.no_assembly_stl,frames=args.frames,hull_wall=args.hull_wall,
        frame_interval=args.frame_interval,frame_coefficient=args.frame_coefficient,
        make_shell=args.make_shell,shell_pitch=args.shell_pitch,
        simplify_faces=args.simplify_faces,simplify_tolerance=args.simplify_tolerance),resume=args.resume)
