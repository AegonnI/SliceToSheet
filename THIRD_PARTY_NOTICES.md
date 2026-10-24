# Third-party software

SliceToSheet 1.1.0 uses Python and third-party libraries for its desktop UI,
3D rendering, solid geometry, CAD import, numerical operations and exports.
Each component and its dependencies retain their own copyright and license.
The versions selected for this release are pinned in `requirements.txt` and
`requirements-build.txt`; installed dependency metadata records transitive versions.

The portable distribution includes available license and notice files under
`licenses/` and package metadata under `_internal/`. Preserve these directories
when redistributing the portable folder.

| Component | Source or license information |
|---|---|
| Python | [Python source and licenses](https://www.python.org/downloads/source/) |
| PySide6, Shiboken6 and Qt | [Qt for Python licensing](https://doc.qt.io/qtforpython-6/licenses.html), [source](https://code.qt.io/pyside/pyside-setup.git/) |
| PyVista | [MIT license and source](https://github.com/pyvista/pyvista/blob/main/LICENSE) |
| PyVistaQt | [MIT license and source](https://github.com/pyvista/pyvistaqt/blob/main/LICENSE) |
| VTK | [Copyright and license](https://github.com/Kitware/VTK/blob/master/Copyright.txt) |
| NumPy | [Source](https://github.com/numpy/numpy) |
| SciPy | [Source](https://github.com/scipy/scipy) |
| Matplotlib | [Source](https://github.com/matplotlib/matplotlib) |
| Trimesh | [Source](https://github.com/mikedh/trimesh) |
| NetworkX | [Source](https://github.com/networkx/networkx) |
| Manifold | [Source](https://github.com/elalish/manifold) |
| fast-simplification | [Source](https://github.com/pyvista/fast-simplification) |
| ezdxf | [MIT license and source](https://github.com/mozman/ezdxf/blob/master/LICENSE) |
| Open3D | [Source](https://github.com/isl-org/Open3D) |
| scikit-image | [Source](https://github.com/scikit-image/scikit-image) |
| cadquery-ocp | [Binding source](https://github.com/CadQuery/OCP) |
| Open CASCADE Technology | [License information](https://occt3d.com/open-cascade-technology/index.html), [source](https://github.com/Open-Cascade-SAS/OCCT) |
| PyInstaller and hooks | [Source](https://github.com/pyinstaller/pyinstaller), [hooks](https://github.com/pyinstaller/pyinstaller-hooks-contrib) |

Qt for Python and Qt include components governed by their applicable LGPL,
commercial or separate third-party terms; consult the notices for the actual
bundled modules. Their shared libraries are supplied as separate files in
the portable distribution.

Open CASCADE Technology is distributed under LGPL 2.1 with the Open CASCADE
exception. See the upstream license and exception corresponding to the
bundled version. The CAD binding and CAD kernel have separate notices.

Other dependencies and third-party code within the listed libraries may have
additional notices in the bundled metadata. This overview does not replace
those license texts.

No separate license grant for the application's own source code is implied
by these third-party notices.
