"""Two-layer frames made by a metric inward offset of an enclosed cavity.

The original shell is only unioned with material inside its cavity. Sampling
is local to one frame; its two longitudinal boundaries are exact cutting
planes. Open3D/scikit-image are loaded only when frames are requested.
"""
import numpy as np
import trimesh
import manifold3d


def is_frame_layer(index, cfg):
    number = index + 1
    return cfg.frames and number >= cfg.frame_interval and number % cfg.frame_interval < 2


class FrameBuilder:
    def __init__(self, mesh, cfg, cancel=lambda: None, log=print):
        self.amount = (getattr(cfg, 'frame_coefficient', 2.0) - 1.0) * cfg.hull_wall
        self.noop = self.amount == 0
        if self.noop:
            return
        if self.amount < 0:
            raise ValueError('Коэффициент шпангоута должен быть не меньше 1.')
        from slicer import as_solid
        try:
            import open3d as o3d
            from skimage.measure import marching_cubes
        except ImportError as error:
            raise ValueError('Для шпангоутов нужны Open3D и scikit-image. '
                             'Запустите START_WINDOWS.bat для установки зависимостей.') from error
        self.o3d = o3d
        self.marching_cubes = marching_cubes
        self.cfg, self.cancel, self.log = cfg, cancel, log
        self.length = float(mesh.bounds[1, 2])
        self.cached_start, self.cached_frame = None, None
        cancel()
        log('Шпангоуты: поиск внутренней полости корпуса…')
        cavities = []
        for component in mesh.split(only_watertight=False, repair=False):
            cancel()
            if component.volume < 0:
                if not component.is_watertight or not component.is_winding_consistent:
                    raise ValueError('Внутренняя полость должна быть замкнутой и согласованно ориентированной.')
                component.invert()
                cavities.append(component)
        mesh._cache.clear()
        if not cavities:
            raise ValueError('Шпангоуты: замкнутая внутренняя полость не найдена. '
                             'Выберите полый корпус (например, hull_hollow.stl). '
                             'Открытые сверху корпуса этим режимом пока не поддерживаются.')
        cavity = trimesh.util.concatenate(cavities)
        cavities.clear()
        self.cavity = as_solid(cavity)
        if self.cavity.status() != manifold3d.Error.NoError:
            raise ValueError('Не удалось преобразовать внутреннюю полость в объём.')
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(cavity.vertices, dtype=np.float32)),
                                 o3d.core.Tensor(np.asarray(cavity.faces, dtype=np.uint32)))
        self.pitch = min(2.0, self.amount / 8, cfg.thickness / 4)
        log('Шпангоуты: слои %d–%d, %d–%d…; стенка %.3g → %.3g мм внутрь.' %
            (cfg.frame_interval, cfg.frame_interval+1, 2*cfg.frame_interval,
             2*cfg.frame_interval+1, cfg.hull_wall, cfg.hull_wall+self.amount))

    def distance(self, points):
        points = np.asarray(points, dtype=np.float32)
        result = np.empty(len(points), dtype=np.float32)
        for start in range(0, len(points), 100000):
            self.cancel()
            batch = self.o3d.core.Tensor(points[start:start+100000])
            result[start:start+100000] = -self.scene.compute_signed_distance(batch, nsamples=3).numpy()
        return result

    def inset(self, local_cavity, low, high):
        """Shrink the global cavity, extracting only a padded local slab."""
        from scipy.ndimage import map_coordinates
        from slicer import as_solid
        amount, pitch = self.amount, self.pitch
        bounds = np.asarray(local_cavity.bounding_box()).reshape(2, 3)
        lower, upper = bounds[0].copy(), bounds[1].copy()
        lower[2], upper[2] = low, high
        lower -= 2*pitch
        upper += 2*pitch
        shape = np.ceil((upper-lower)/pitch).astype(int) + 1
        if np.prod(shape, dtype=np.float64) > 60000000:
            raise ValueError('Слишком большая сетка шпангоута. Проверьте масштаб модели и толщину стенки.')
        axes = [lower[i] + np.arange(shape[i])*pitch for i in range(3)]
        coarse_pitch = max(8.0, 4*pitch)
        coarse_axes = [lower[i] + np.arange(int(np.ceil((axes[i][-1]-lower[i])/coarse_pitch))+1)*coarse_pitch
                       for i in range(3)]
        # Query in strips, keeping both memory and cancellation latency bounded.
        def points_for(x, y, z):
            return np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
        coarse = np.empty(tuple(len(a) for a in coarse_axes), dtype=np.float32)
        for i, x in enumerate(coarse_axes[0]):
            coarse[i] = self.distance(points_for([x], *coarse_axes[1:])).reshape(coarse.shape[1:])
        field = np.empty(tuple(shape), dtype=np.float32)
        band = np.sqrt(3)*(coarse_pitch+pitch)
        for i, x in enumerate(axes[0]):
            self.cancel()
            points = points_for([x], *axes[1:])
            values = map_coordinates(coarse, ((points-lower)/coarse_pitch).T,
                                     order=1, mode='nearest', prefilter=False)
            near = np.abs(values-amount) <= band
            values[near] = self.distance(points[near])
            field[i] = (values-amount).reshape(tuple(shape[1:]))
            if i % 100 == 0:
                self.log('Шпангоут: внутренняя поверхность, %d/%d сечений сетки.' % (i+1, shape[0]))
        # Artificial caps lie outside the frame. Exact trim planes remove them.
        field[0, :, :] = field[-1, :, :] = -pitch
        field[:, 0, :] = field[:, -1, :] = -pitch
        field[:, :, 0] = field[:, :, -1] = -pitch
        if field.max() <= 0:
            return manifold3d.Manifold()
        self.cancel()
        vertices, faces, _, _ = self.marching_cubes(field, level=0, spacing=(pitch,)*3,
                                                    allow_degenerate=False)
        del field
        vertices = vertices.astype(np.float64) + lower
        inner = trimesh.Trimesh(vertices, faces, process=True)
        if inner.volume < 0:
            inner.invert()
        if not inner.is_watertight or not inner.is_volume:
            raise ValueError('Не удалось построить замкнутую внутреннюю поверхность шпангоута.')
        # Check the new surface in the actual two-layer band, excluding caps.
        selected = vertices[(vertices[:, 2] >= low) & (vertices[:, 2] <= high)]
        error = float(np.max(np.abs(self.distance(selected)-amount))) if len(selected) else 0.0
        if error > max(0.1, pitch):
            raise ValueError('Погрешность утолщения шпангоута %.3g мм превышает допуск %.3g мм.' %
                             (error, max(0.1, pitch)))
        self.log('Шпангоут: проверка отступа внутренней поверхности, погрешность до %.3g мм.' % error)
        solid = as_solid(inner)
        if solid.status() != manifold3d.Error.NoError:
            raise ValueError('Ошибка объёмной геометрии шпангоута.')
        solid = solid.trim_by_plane([0, 0, 1], low)
        return solid if solid.is_empty() else solid.trim_by_plane([0, 0, -1], -high)

    def augment(self, slab, index):
        if self.noop:
            return slab
        if not is_frame_layer(index, self.cfg):
            return slab
        start = ((index+1)//self.cfg.frame_interval)*self.cfg.frame_interval-1
        if start != self.cached_start:
            self.cancel()
            low = start*self.cfg.thickness
            high = min((start+2)*self.cfg.thickness, self.length)
            self.log('Шпангоут: слои %d–%d.' % (start+1, min(start+2, int(np.ceil(self.length/self.cfg.thickness)))))
            cavity = self.cavity.trim_by_plane([0, 0, 1], low)
            if not cavity.is_empty():
                cavity = cavity.trim_by_plane([0, 0, -1], -high)
            self.cached_frame = cavity if cavity.is_empty() else cavity - self.inset(cavity, low, high)
            self.cached_start = start
        self.cancel()
        if self.cached_frame.is_empty():
            return slab
        frame = self.cached_frame.trim_by_plane([0, 0, 1], index*self.cfg.thickness)
        if frame.is_empty():
            return slab
        frame = frame.trim_by_plane([0, 0, -1], -min((index+1)*self.cfg.thickness, self.length))
        if frame.is_empty():
            return slab
        result = slab + frame
        if result.status() != manifold3d.Error.NoError:
            raise ValueError('Не удалось объединить шпангоут с корпусом.')
        return result
