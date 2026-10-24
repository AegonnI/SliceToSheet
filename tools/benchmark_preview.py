r"""Compare camera redraws of the same 60,000 triangles in Matplotlib and VTK.

Run with the application's runtime, for example:
  .venv-app\Scripts\python.exe tools/benchmark_preview.py

This measures warmed camera renders, not loading, decimation, slicing or app
startup. VTK waits for GPU completion so command submission alone is not timed.
"""
import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh


def benchmark_mesh():
    longitude_count, ring_count = 200, 150
    phi = np.linspace(0, 2*np.pi, longitude_count, endpoint=False)
    theta = np.linspace(0, np.pi, ring_count+2)[1:-1]
    points = np.stack((np.sin(theta)[:,None]*np.cos(phi),
                       np.sin(theta)[:,None]*np.sin(phi),
                       np.broadcast_to(np.cos(theta)[:,None], (ring_count,longitude_count))), axis=-1)
    vertices = np.vstack((points.reshape(-1,3), [0,0,1], [0,0,-1]))*50
    first = np.arange((ring_count-1)*longitude_count).reshape(-1,longitude_count)
    next_column = np.roll(first,-1,axis=1)
    faces = [np.column_stack((first.ravel(), (first+longitude_count).ravel(), next_column.ravel())),
             np.column_stack((next_column.ravel(), (first+longitude_count).ravel(),
                              (next_column+longitude_count).ravel()))]
    top, bottom = len(vertices)-2, len(vertices)-1
    columns = np.arange(longitude_count)
    following = np.roll(columns,-1)
    faces.append(np.column_stack((np.full(longitude_count,top), columns, following)))
    last = (ring_count-1)*longitude_count
    faces.append(np.column_stack((np.full(longitude_count,bottom), last+following, last+columns)))
    mesh = trimesh.Trimesh(vertices, np.vstack(faces), process=False)
    if mesh.volume < 0:
        mesh.invert()
    assert len(mesh.faces) == 60000
    assert mesh.is_watertight
    return mesh


def timings(render, angles, warmup):
    for angle in angles[:warmup]:
        render(angle)
    samples = []
    for angle in angles:
        start = time.perf_counter()
        render(angle)
        samples.append(1000*(time.perf_counter()-start))
    return dict(samples_ms=samples, median_ms=statistics.median(samples),
                minimum_ms=min(samples), maximum_ms=max(samples),
                p95_ms=float(np.percentile(samples,95)))


def matplotlib_render(mesh, dimensions, angles, warmup):
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    figure = Figure(figsize=(dimensions[0]/100,dimensions[1]/100), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(projection='3d')
    axes.set_position([0,0,1,1])
    axes.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], facecolor='#62a9c6',
                                          edgecolor='none', antialiased=False))
    axes.set(xlim=(-55,55), ylim=(-55,55), zlim=(-55,55))
    axes.set_box_aspect((1,1,1))
    axes.set_axis_off()
    def render(angle):
        axes.view_init(elev=30, azim=angle)
        canvas.draw()
    result = timings(render, angles, warmup)
    result.update(version=matplotlib.__version__, renderer='Agg CPU rasterizer',
                  timed_operation='axes.view_init + full canvas.draw')
    return result


def vtk_render(mesh, dimensions, angles, warmup):
    import pyvista as pv
    import vtk
    faces = np.column_stack((np.full(len(mesh.faces),3),mesh.faces)).ravel()
    plotter = pv.Plotter(off_screen=True, window_size=dimensions, lighting='none')
    plotter.ren_win.SetMultiSamples(0)
    plotter.add_mesh(pv.PolyData(mesh.vertices,faces), color='#62a9c6', show_edges=False,
                     lighting=False, render=False)
    plotter.camera.focal_point = (0,0,0)
    plotter.camera.up = (0,0,1)
    plotter.camera.view_angle = 30
    plotter.camera.clipping_range = (.1,1000)
    elevation = math.radians(30)
    def render(angle):
        azimuth = math.radians(angle)
        plotter.camera.position = (220*math.cos(elevation)*math.cos(azimuth),
                                   220*math.cos(elevation)*math.sin(azimuth),
                                   220*math.sin(elevation))
        plotter.render()
        plotter.ren_win.WaitForCompletion()
    # Initialize the OpenGL context before recording any warmed render.
    plotter.show(auto_close=False, interactive=False)
    try:
        result = timings(render, angles, warmup)
        capabilities = plotter.ren_win.ReportCapabilities()
        graphics = [line.strip() for line in capabilities.splitlines()
                    if any(word in line.lower() for word in ('vendor','renderer','version'))
                    and not line.lstrip().startswith('GL_')]
        result.update(pyvista_version=pv.__version__, vtk_version=vtk.vtkVersion.GetVTKVersion(),
                      renderer=plotter.ren_win.GetClassName(), graphics=graphics[:12],
                      timed_operation='camera.position + render + GPU WaitForCompletion')
        return result
    finally:
        plotter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'build'/'qa'/'preview-benchmark.json')
    parser.add_argument('--width',type=int,default=800)
    parser.add_argument('--height',type=int,default=600)
    parser.add_argument('--angles',type=int,default=10)
    parser.add_argument('--warmup',type=int,default=3)
    args = parser.parse_args()
    mesh = benchmark_mesh()
    digest = hashlib.sha256(mesh.vertices.tobytes()+mesh.faces.tobytes()).hexdigest()
    angles = np.linspace(0,360,args.angles,endpoint=False).tolist()
    dimensions = [args.width,args.height]
    old = matplotlib_render(mesh,dimensions,angles,args.warmup)
    new = vtk_render(mesh,dimensions,angles,args.warmup)
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                  python=platform.python_version(), numpy=np.__version__,
                  mesh=dict(triangles=len(mesh.faces),vertices=len(mesh.vertices),sha256=digest,
                            geometry='UV sphere, 150 rings by 200 longitudes plus two poles',
                            identical_mesh_for_both=True),
                  dimensions_px=dimensions, angles_degrees=angles, warmup_renders=args.warmup,
                  matplotlib=old, vtk=new, median_speedup=old['median_ms']/new['median_ms'],
                  caveats=[
                      'Measures warmed complete camera renders only; excludes loading, preparation, decimation and slicing.',
                      'Same geometry, dimensions, solid color and disabled edges; lighting and anti-aliasing disabled.',
                      'Matplotlib Agg and VTK OpenGL use different rasterizers and camera projections; output pixels are not identical.',
                      'VTK GPU completion is awaited; timings depend on the actual OpenGL driver, hardware and desktop session.',
                      'One synthetic closed 60k-triangle surface and ten angles do not establish performance for every model.',
                  ])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),matplotlib_median_ms=old['median_ms'],
                          vtk_median_ms=new['median_ms'],median_speedup=report['median_speedup'])))


if __name__ == '__main__':
    main()
