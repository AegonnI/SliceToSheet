"""Cached preview geometry; physical sizing is independent of slicing axis."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import trimesh


def preview_mesh(mesh, face_count=60000):
    if len(mesh.faces) <= face_count:
        return mesh.copy()
    try:
        return mesh.simplify_quadric_decimation(face_count=face_count)
    except (ImportError, ValueError, RuntimeError):
        import pyvista as pv
        faces = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).ravel()
        result = pv.PolyData(np.asarray(mesh.vertices), faces).decimate(
            target_reduction=max(0, 1-face_count/len(mesh.faces)))
        return trimesh.Trimesh(result.points, result.faces.reshape(-1, 4)[:, 1:], process=False)


@dataclass
class PreviewData:
    mesh: object
    extents: np.ndarray
    source_faces: int
    source_key: tuple


def source_key(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


def prepare_preview(path, face_count=60000):
    from slicer import Settings, load_model
    key = source_key(path)
    mesh, _ = load_model(path, Settings(axis='Z', reverse=False, target_length=0, scale=1))
    return PreviewData(preview_mesh(mesh, face_count), mesh.extents.copy(), len(mesh.faces), key)


def transform_preview(data, scale, axis, reverse=False):
    mesh = data.mesh.copy()
    matrix = trimesh.geometry.align_vectors(np.eye(3)['XYZ'.index(axis)], [0, 0, -1 if reverse else 1])
    matrix[:3, :3] *= scale
    mesh.apply_transform(matrix)
    mesh.apply_translation(-mesh.bounds[0])
    return mesh


def preview_extents(data, scale, axis):
    rotation = trimesh.geometry.align_vectors(np.eye(3)['XYZ'.index(axis)], [0, 0, 1])
    return np.abs(rotation[:3, :3]) @ (data.extents*scale)
