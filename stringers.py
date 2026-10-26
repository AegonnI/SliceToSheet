"""Longitudinal inward ribs, spaced by arc length up both sides of a cavity.

The transverse side distance sets depth; an arc coordinate sets rib
width. Local padded grids keep memory bounded and exact slab/hull clipping
preserves the outside hull and layer boundaries.
"""
import math

import numpy as np
import manifold3d
from scipy.spatial import cKDTree


def side_paths(polygon, up=1):
    """Walk from the middle of the keel to the first top edge on each side."""
    polygon = np.asarray(polygon, dtype=float)
    if len(polygon) < 3:
        return []
    bottom = polygon[:, up].min()
    top = polygon[:, up].max()
    if top-bottom < 1e-6:
        return []
    keel_vertices = polygon[np.isclose(polygon[:, up], bottom, atol=1e-6, rtol=0)]
    keel = (keel_vertices.min(axis=0)+keel_vertices.max(axis=0))/2
    ends = np.roll(polygon, -1, axis=0)
    delta = ends-polygon
    lengths2 = np.einsum('ij,ij->i', delta, delta)
    t = np.clip(np.einsum('ij,ij->i', keel-polygon, delta)/np.maximum(lengths2, 1e-20), 0, 1)
    projected = polygon+t[:, None]*delta
    edge = int(np.argmin(np.linalg.norm(projected-keel, axis=1)))
    keel = projected[edge]
    loop = np.vstack([keel, polygon[(np.arange(len(polygon))+edge+1) % len(polygon)], keel])
    result = []
    for points in (loop, loop[::-1]):
        stop = np.flatnonzero(np.isclose(points[:, up], top, atol=1e-6, rtol=0))[0]
        path = points[:stop+1]
        path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-7]]
        if len(path) >= 2:
            result.append(path)
    return result


class ArcPath:
    def __init__(self, points, pitch):
        length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        self.length = float(length[-1])
        # Every segment is short, so nearby midpoint candidates include the
        # exact nearest segment even on long, originally unsegmented walls.
        samples = np.unique(np.r_[length, np.arange(0, self.length, pitch)])
        self.points = np.column_stack([np.interp(samples, length, points[:, i]) for i in range(2)])
        self.arc = samples
        self.start = self.points[:-1]
        self.delta = np.diff(self.points, axis=0)
        self.squared = np.einsum('ij,ij->i', self.delta, self.delta)
        self.tree = cKDTree(self.start+self.delta/2)

    def interval(self, low, high):
        samples = np.r_[max(0., low), self.arc[(self.arc > low) & (self.arc < high)], min(high, self.length)]
        return np.column_stack([np.interp(samples, self.arc, self.points[:, i]) for i in range(2)])

    def locate(self, points):
        _, index = self.tree.query(points, k=min(8, len(self.start)))
        if index.ndim == 1:
            index = index[:, None]
        delta = self.delta[index]
        t = np.clip(np.einsum('nki,nki->nk', points[:, None, :]-self.start[index], delta)/
                    np.maximum(self.squared[index], 1e-20), 0, 1)
        projected = self.start[index]+t[:, :, None]*delta
        best = np.argmin(np.sum((projected-points[:, None, :])**2, axis=2), axis=1)
        rows = np.arange(len(points))
        segment = index[rows, best]
        arc = self.arc[segment]+t[rows, best]*(self.arc[segment+1]-self.arc[segment])
        distance = np.linalg.norm(projected[rows, best]-points, axis=1)
        return arc, distance

    def coordinate(self, points):
        return self.locate(points)[0]


class StringerBuilder:
    def __init__(self, mesh, cfg, cancel=lambda: None, log=print):
        import open3d as o3d
        from skimage.measure import marching_cubes
        import trimesh
        from slicer import as_solid

        self.cfg, self.cancel, self.log = cfg, cancel, log
        self.o3d, self.marching_cubes = o3d, marching_cubes
        self.pitch = min(2., cfg.stringer_height/8, cfg.stringer_depth/8, cfg.thickness/4)
        self.length = float(mesh.bounds[1, 2])
        cavities = []
        for component in mesh.split(only_watertight=False, repair=False):
            cancel()
            if component.volume < 0:
                if not component.is_watertight or not component.is_winding_consistent:
                    raise ValueError('Стрингеры: внутренняя полость должна быть замкнутой.')
                component.invert()
                cavities.append(component)
        mesh._cache.clear()
        if not cavities:
            raise ValueError('Стрингеры: замкнутая внутренняя полость не найдена. '
                             'Включите создание оболочки или выберите полый STL.')
        cavity = trimesh.util.concatenate(cavities)
        self.cavity = as_solid(cavity)
        self.cavity_bounds = np.asarray(self.cavity.bounding_box()).reshape(2, 3)
        if self.cavity.status() != manifold3d.Error.NoError:
            raise ValueError('Стрингеры: некорректный объём полости.')
        # Keep a small overlap into the existing wall. Clipping ribs to the
        # cavity alone can leave coincident surfaces separated by roundoff
        # on sloping triangles. The filled hull clips only the outside.
        self.envelope = as_solid(mesh)+self.cavity
        if self.envelope.status() != manifold3d.Error.NoError:
            raise ValueError('Стрингеры: не удалось восстановить наружный объём корпуса.')
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(cavity.vertices, dtype=np.float32)),
                                 o3d.core.Tensor(np.asarray(cavity.faces, dtype=np.uint32)))
        self.cache = {}
        log('Стрингеры вдоль оси нарезки: шаг по внутреннему борту %.3g мм, '
            'выступ внутрь %.3g мм, высота %.3g мм; сетка %.3g мм.' %
            (cfg.stringer_spacing, cfg.stringer_depth, cfg.stringer_height, self.pitch))

    def profiles(self, station):
        if station not in self.cache:
            self.cancel()
            polygons = self.cavity.slice(station*self.pitch).to_polygons()
            polygons = sorted(polygons, key=lambda p: tuple(np.min(p, axis=0)))
            paths = {}
            for number, polygon in enumerate(polygons):
                # A cavity cross-section can itself contain islands. Those
                # clockwise contours are not the two sides of the hull.
                area = np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)-
                              polygon[:, 1]*np.roll(polygon[:, 0], -1))
                if area <= 0:
                    continue
                for side, points in enumerate(side_paths(polygon, 'XY'.index(self.cfg.stringer_up))):
                    paths[number, side] = ArcPath(points, self.pitch)
            self.cache[station] = paths
        return self.cache[station]

    def augment(self, slab, index):
        from slicer import as_solid
        import trimesh

        self.cancel()
        p, cfg = self.pitch, self.cfg
        low = index*cfg.thickness
        high = min((index+1)*cfg.thickness, self.length)
        if low >= self.cavity_bounds[1, 2]-1e-6 or high <= self.cavity_bounds[0, 2]+1e-6:
            return slab
        first, last = math.floor(low/p)-2, math.ceil(high/p)+2
        stations = range(first, last+1)
        profiles = {s: self.profiles(s) for s in stations}
        self.cache = {s: value for s, value in self.cache.items() if s >= last-4}
        candidates = {}
        half = cfg.stringer_height/2
        for station, paths in profiles.items():
            for key, path in paths.items():
                for rib in range(1, int((path.length-half)/cfg.stringer_spacing)+1):
                    center = rib*cfg.stringer_spacing
                    candidates.setdefault((*key, rib), []).append(path.interval(center-half, center+half))
        for key, boundaries in sorted(candidates.items()):
            self.cancel()
            center = key[2]*cfg.stringer_spacing
            boundary = np.vstack(boundaries)
            # A metric depth can extend in any transverse direction.
            lower = np.floor((boundary.min(axis=0)-cfg.stringer_depth-3*p)/p)*p
            upper = np.ceil((boundary.max(axis=0)+cfg.stringer_depth+3*p)/p)*p
            shape = np.ceil((upper-lower)/p).astype(int)+1
            if np.prod(shape, dtype=float)*len(profiles) > 12000000:
                raise ValueError('Слишком большая сетка стрингера. Уменьшите толщину слоя или выступ.')
            x, y = [lower[i]+np.arange(shape[i])*p for i in range(2)]
            xy = np.stack(np.meshgrid(x, y, indexing='ij'), axis=-1).reshape(-1, 2)
            field = np.full((*shape, len(profiles)), -p, dtype=np.float32)
            for j, station in enumerate(stations):
                self.cancel()
                path = profiles[station].get(key[:2])
                if path is None or center+half > path.length:
                    continue
                arc, side_distance = path.locate(xy)
                stripe = half-np.abs(arc-center)
                # Only query the narrow strip; the rest stays outside.
                near = stripe >= -2*p
                near_indices = np.flatnonzero(near)
                points = np.column_stack([xy[near], np.full(np.count_nonzero(near), station*p)])
                values = np.full(len(xy), -p, dtype=np.float32)
                for start in range(0, len(points), 100000):
                    self.cancel()
                    batch = self.o3d.core.Tensor(np.asarray(points[start:start+100000], dtype=np.float32))
                    distance = -self.scene.compute_signed_distance(batch, nsamples=3).numpy()
                    indices = near_indices[start:start+100000]
                    # Overlap the wall, then clip by the exact hull. This
                    # prevents an interpolated zero surface leaving a gap.
                    values[indices] = np.minimum.reduce([distance+2*p, cfg.stringer_depth-side_distance[indices], stripe[indices]])
                field[:, :, j] = values.reshape(tuple(shape))
            field[0, :, :] = field[-1, :, :] = -p
            field[:, 0, :] = field[:, -1, :] = -p
            field[:, :, 0] = field[:, :, -1] = -p
            if field.max() <= 0:
                continue
            vertices, faces, _, _ = self.marching_cubes(field, 0, spacing=(p,)*3, allow_degenerate=False)
            vertices = vertices.astype(float)+[lower[0], lower[1], first*p]
            mesh = trimesh.Trimesh(vertices, faces, process=True)
            if mesh.volume < 0:
                mesh.invert()
            rib = as_solid(mesh)
            if rib.status() != manifold3d.Error.NoError:
                raise ValueError('Не удалось построить замкнутый стрингер.')
            rib = rib ^ self.envelope
            if rib.is_empty():
                continue
            rib = rib.trim_by_plane([0, 0, 1], low)
            if rib.is_empty():
                continue
            rib = rib.trim_by_plane([0, 0, -1], -high)
            if rib.is_empty():
                continue
            slab = slab+rib
            if slab.status() != manifold3d.Error.NoError:
                raise ValueError('Не удалось объединить стрингер с корпусом: %s (слой %d).' % (slab.status(), index+1))
        return slab
