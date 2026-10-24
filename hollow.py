"""Metric inward offset of a closed STL. Keeps the original outer triangles.

Positive interior distance is sampled with Open3D's triangle BVH. A coarse
distance grid only rejects points safely away from the offset; all fine grid
vertices that can contribute to the extracted surface get exact BVH queries.
Slab extraction bounds memory independently of the length of the vessel.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes


def scene_for(mesh):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
    return scene


def distance(scene, points, signed=True, batch=200000, cancel=lambda: None):
    import open3d as o3d
    points = np.asarray(points, dtype=np.float32)
    result = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), batch):
        cancel()
        p = o3d.core.Tensor(points[start:start + batch])
        if signed:
            result[start:start + batch] = -scene.compute_signed_distance(p, nsamples=3).numpy()
        else:
            result[start:start + batch] = scene.compute_distance(p).numpy()
    return result


def grid_points(axes):
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)


def prepare_mesh(path, length=8000., axis='X'):
    if not math.isfinite(length) or length <= 0 or axis not in 'XYZ' or len(axis) != 1:
        raise ValueError('Length must be positive and axis X, Y or Z.')
    mesh = trimesh.load(str(path), force='mesh', process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('Empty mesh.')
    if not np.isfinite(mesh.vertices).all() or not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError('Input must be finite, closed and consistently oriented.')
    if mesh.volume < 0:
        mesh.invert()
    if not mesh.is_volume:
        raise ValueError('Input is not a valid volume.')
    span = float(mesh.extents['XYZ'.index(axis)])
    if span <= 0:
        raise ValueError('Zero length.')
    factor = length / span
    source_min = mesh.bounds[0].copy()
    mesh.apply_translation(-source_min)
    mesh.apply_scale(factor)
    return mesh, {'scale': factor, 'source_min': source_min.tolist(),
                  'axis': axis, 'length_mm': length}


def make_inner(mesh, wall=50., pitch=3., coarse_pitch=20., log=print, diagnostic_path=None, cache_dir=None,
               cancel=lambda: None):
    if not all(math.isfinite(x) and x > 0 for x in (wall, pitch, coarse_pitch)):
        raise ValueError('Wall and grid pitches must be finite and positive.')
    if pitch > wall / 4:
        raise ValueError('Fine grid pitch must be at most one quarter of the wall.')
    if coarse_pitch < pitch:
        raise ValueError('Coarse grid pitch must be at least the fine pitch.')
    cancel()
    scene = scene_for(mesh)
    cancel()
    origin = mesh.bounds[0] - coarse_pitch
    coarse_axes = [origin[i] + np.arange(int(np.ceil((mesh.bounds[1, i] - origin[i]) /
                   coarse_pitch)) + 2) * coarse_pitch for i in range(3)]
    shape = tuple(len(a) for a in coarse_axes)
    if np.prod(shape) > 100000000:
        raise ValueError('Coarse grid too large; increase coarse pitch.')
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(exist_ok=True)
    coarse_file = cache_dir / 'coarse.npy' if cache_dir else None
    if coarse_file and coarse_file.exists():
        field = np.load(coarse_file)
        log('Restored coarse distance grid')
    else:
        field = np.empty(shape, dtype=np.float32)
        log(f'Coarse distance grid: {shape}')
        for a in range(0, shape[0], 8):
            cancel()
            axes = [coarse_axes[0][a:a+8], *coarse_axes[1:]]
            field[a:a+8] = distance(scene, grid_points(axes), cancel=cancel).reshape(len(axes[0]), shape[1], shape[2])
            if a % 32 == 0 or a + 8 >= shape[0]:
                log(f'Coarse distances: {min(a+8, shape[0])}/{shape[0]} planes')
        if coarse_file:
            with coarse_file.with_suffix('.tmp').open('wb') as stream:
                np.save(stream, field)
            os.replace(coarse_file.with_suffix('.tmp'), coarse_file)
    if field.max() + np.sqrt(3) * coarse_pitch < wall:
        raise ValueError('No room for a cavity with this wall thickness.')
    fine_axes = [origin[i] + np.arange(int(np.ceil((mesh.bounds[1, i] - origin[i]) /
                 pitch)) + 2) * pitch for i in range(3)]
    # Signed distance is 1-Lipschitz. The diagonal of a coarse cell bounds
    # trilinear interpolation error; extra fine-cell margin protects extraction.
    band = np.sqrt(3) * (coarse_pitch + pitch)
    vertices, faces = [], []
    count = 0
    slab_cells = max(1, min(16, 2000000 // (len(fine_axes[1]) * len(fine_axes[2]))))
    for a in range(0, len(fine_axes[0]) - 1, slab_cells):
        cancel()
        b = min(a + slab_cells, len(fine_axes[0]) - 1)
        block_file = cache_dir / f'block_{a:06d}.npz' if cache_dir else None
        if block_file and block_file.exists():
            with np.load(block_file) as saved:
                v, f = saved['vertices'], saved['faces']
        else:
            axes = [fine_axes[0][a:b+1], *fine_axes[1:]]
            points = grid_points(axes)
            values = map_coordinates(field, ((points - origin) / coarse_pitch).T, order=1,
                                     mode='nearest', prefilter=False)
            near = np.abs(values - wall) <= band
            values[near] = distance(scene, points[near], cancel=cancel)
            volume = values.reshape(len(axes[0]), len(axes[1]), len(axes[2]))
            v, f = np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
            if volume.min() < wall < volume.max():
                v, f, _, _ = marching_cubes(volume, level=wall, spacing=(pitch,) * 3,
                                           allow_degenerate=False)
                v = v.astype(np.float64)
                v += [axes[0][0], axes[1][0], axes[2][0]]
            if block_file:
                with block_file.with_suffix('.tmp').open('wb') as stream:
                    np.savez(stream, vertices=v, faces=f)
                os.replace(block_file.with_suffix('.tmp'), block_file)
        if len(v):
            vertices.append(v)
            faces.append(f + count)
            count += len(v)
        log(f'Inner surface: {b}/{len(fine_axes[0])-1} longitudinal grid cells')
    if not vertices:
        raise ValueError('No cavity found. Reduce wall or grid pitch.')
    inner = trimesh.Trimesh(np.vstack(vertices), np.vstack(faces), process=True)
    # Simplify only the new inner surface, in double precision and by 0.01 mm.
    # This also prevents dense-grid adjacency caches from dominating memory.
    vertices.clear(); faces.clear()
    cancel()
    inner = simplify_inner(inner)
    cancel()
    if inner.volume < 0:
        inner.invert()
    if not inner.is_watertight or not inner.is_volume:
        if diagnostic_path is not None:
            np.savez_compressed(diagnostic_path, vertices=inner.vertices, faces=inner.faces)
        raise ValueError('Extracted cavity is not a closed volume; export stopped.')
    return inner, scene


def shell_from_inner(outer, inner):
    cavity = inner.copy()
    cavity.invert()
    shell = trimesh.util.concatenate([outer, cavity])
    if not shell.is_watertight or not shell.is_winding_consistent or not shell.is_volume:
        raise ValueError('Invalid shell topology.')
    import manifold3d
    solid = manifold3d.Manifold(manifold3d.Mesh(
        vert_properties=np.asarray(shell.vertices, dtype=np.float32),
        tri_verts=np.asarray(shell.faces, dtype=np.uint32)))
    if solid.status() != manifold3d.Error.NoError:
        raise ValueError(f'Slicer solid conversion failed: {solid.status()}')
    return shell


def simplify_inner(inner, tolerance=.01):
    """Remove sub-STL-resolution edges in double precision before quantization."""
    import manifold3d
    solid = manifold3d.Manifold(manifold3d.Mesh64(
        vert_properties=np.array(inner.vertices, dtype=np.float64, order='C', copy=True),
        tri_verts=np.array(inner.faces, dtype=np.uint64, order='C', copy=True)))
    if solid.status() != manifold3d.Error.NoError:
        raise ValueError(f'Internal surface topology rejected: {solid.status()}')
    data = solid.simplify(tolerance).to_mesh64()
    return trimesh.Trimesh(data.vert_properties[:, :3], data.tri_verts, process=False)


def stl_safe_inner(inner, log=print, cancel=lambda: None):
    """Separate touching cavity sheets by hundredths of a mm, then verify STL.

    A native manifold can represent two sheets at the same coordinates with
    different vertex IDs; STL cannot. Move only coincident vertices inward,
    keeping this tiny regularization covered by the final distance check.
    """
    vertices = np.asarray(inner.vertices).copy()
    normals = None
    for attempt in range(8):
        cancel()
        quantized = vertices.astype(np.float32)
        _, inverse, counts = np.unique(quantized, axis=0, return_inverse=True, return_counts=True)
        coincident = counts[inverse] > 1
        candidate = trimesh.Trimesh(quantized, inner.faces, process=True)
        f = candidate.faces
        distinct = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
        candidate.update_faces(candidate.unique_faces() & distinct)
        candidate.remove_unreferenced_vertices()
        if candidate.is_watertight and candidate.is_volume:
            return candidate
        if not np.any(coincident):
            break
        if normals is None:
            normals = inner.vertex_normals
        log(f'STL contact regularization: {coincident.sum()} coincident vertices, step 0.01 mm')
        vertices[coincident] -= normals[coincident] * .01
    raise ValueError('Cannot represent the inner surface as a closed STL without contact.')


def validate_inner(inner, scene, wall, tolerance=2., samples=100000, raise_on_failure=True,
                   cancel=lambda: None):
    # Check every vertex plus reproducible samples inside faces. This is a
    # numerical check, not a proof of a global minimum thickness.
    rng = np.random.default_rng(726)
    ids = rng.choice(len(inner.faces), size=samples, p=inner.area_faces / inner.area)
    triangles = inner.triangles[ids]
    uv = rng.random((samples, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    probes = triangles[:, 0] + uv[:, :1] * (triangles[:, 1] - triangles[:, 0]) + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])
    d = distance(scene, np.vstack([inner.vertices, probes]), cancel=cancel)
    error = np.abs(d - wall)
    report = {'checked_points': len(d), 'min_mm': float(d.min()), 'max_mm': float(d.max()),
              'mean_mm': float(d.mean()), 'max_abs_error_mm': float(error.max()),
              'p99_abs_error_mm': float(np.quantile(error, .99)),
              'tolerance_mm': tolerance, 'passed': bool(error.max() <= tolerance)}
    if not report['passed'] and raise_on_failure:
        raise ValueError('Thickness check failed; use a finer grid: ' + json.dumps(report))
    return report


def run(source, output, length=8000., axis='X', wall=50., pitch=3., tolerance=2., log=print, resume=False):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Tolerance must be finite and positive.')
    output = Path(output)
    job = {'version': 1, 'sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
           'length': length, 'axis': axis, 'wall': wall, 'pitch': pitch, 'tolerance': tolerance}
    if resume:
        if json.loads((output / 'job.json').read_text(encoding='utf-8')) != job:
            raise ValueError('Source or parameters differ from the saved calculation.')
        if (output / 'report.json').exists():
            raise ValueError('This calculation is already complete.')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'job.json').write_text(json.dumps(job, indent=2), encoding='utf-8')
    started = time.time()
    outer, transform = prepare_mesh(source, length, axis)
    log(f'Outer dimensions in mm: {outer.extents.tolist()}')
    inner, scene = make_inner(outer, wall, pitch, max(20., pitch * 4), log,
                             output / 'failed_inner.npz', output / 'blocks')
    # Keep an explicitly labelled diagnostic checkpoint if validation fails.
    inner.export(output / 'cavity_candidate.stl')
    return finish(source, output, outer, inner, scene, transform, wall, pitch,
                  tolerance, started, log)


def finish(source, output, outer, inner, scene, transform, wall, pitch, tolerance, started, log=print):
    output = Path(output)
    # Normalize a checkpoint too: binary STL can collapse tiny triangles.
    if not inner.is_watertight:
        f = inner.faces
        distinct = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
        inner.update_faces(inner.unique_faces() & distinct)
        inner.remove_unreferenced_vertices()
    inner = simplify_inner(inner)
    log(f'Inner mesh after 0.01 mm simplification: {len(inner.faces)} triangles')
    inner = stl_safe_inner(inner, log)
    inner.export(output / 'cavity_candidate.stl')
    check = validate_inner(inner, scene, wall, tolerance)
    log('Thickness check: ' + json.dumps(check))
    shell = shell_from_inner(outer, inner)
    target = output / 'hull_hollow.stl'
    shell.export(target)
    # STL is float32: verify the actual deliverable after serialization.
    reloaded = trimesh.load(target, force='mesh')
    if not reloaded.is_volume or not reloaded.is_winding_consistent:
        raise ValueError('STL round-trip failed.')
    if not np.isclose(reloaded.volume, shell.volume, rtol=1e-5):
        raise ValueError('STL round-trip changed shell volume.')
    inner.export(output / 'cavity.stl')
    report = {'source': str(Path(source).resolve()),
              'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              'transform': transform, 'extents_mm': outer.extents.tolist(),
              'wall_mm': wall, 'grid_pitch_mm': pitch, 'closed_cavity': True,
              'outer_faces': len(outer.faces), 'inner_faces': len(inner.faces),
              'outer_volume_m3': float(outer.volume / 1e9),
              'cavity_volume_m3': float(inner.volume / 1e9),
              'material_volume_m3': float(shell.volume / 1e9),
              'retained_volume_percent': float(100 * shell.volume / outer.volume),
              'watertight': bool(reloaded.is_watertight),
              'winding_consistent': bool(reloaded.is_winding_consistent),
              'thickness_check': check, 'elapsed_seconds': time.time() - started,
              'limitations': ['No openings added; thin regions may remain solid.',
                             'Thickness tested from inner surface to original outer surface.',
                             'No structural strength or global self-intersection certification.']}
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    log(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--length', type=float, default=8000.)
    parser.add_argument('--axis', choices=['X', 'Y', 'Z'], default='X')
    parser.add_argument('--wall', type=float, default=50.)
    parser.add_argument('--pitch', type=float, default=3.)
    parser.add_argument('--tolerance', type=float, default=2.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.source, args.output, args.length, args.axis, args.wall, args.pitch, args.tolerance, resume=args.resume)
