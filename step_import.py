"""Import STEP solids as a millimetre triangle mesh; OCP is loaded on demand."""
import tempfile
import shutil
from pathlib import Path

import trimesh


def load_step(path):
    try:
        from OCP.STEPControl import STEPControl_Reader
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.StlAPI import StlAPI_Writer
    except ImportError as error:
        raise ValueError(
            'Для STEP/STP нужен cadquery-ocp. Установите зависимости из requirements.txt '
            'или заново распакуйте полный ZIP-релиз SliceToSheet.') from error

    # Use simple temporary names for native file APIs, including Unicode sources.
    with tempfile.TemporaryDirectory(prefix='step_import_') as folder:
        source = Path(folder) / 'input.step'
        shutil.copyfile(path, source)
        reader = STEPControl_Reader()
        if reader.ReadFile(str(source)) != IFSelect_RetDone:
            raise ValueError('Не удалось прочитать STEP/STP: файл повреждён или имеет неверный формат.')
        reader.SetSystemLengthUnit(1.0)  # OCCT internal length unit: millimetres.
        if reader.TransferRoots() == 0:
            raise ValueError('STEP/STP не содержит геометрии для нарезки.')
        shape = reader.OneShape()
        if shape.IsNull():
            raise ValueError('STEP/STP не содержит геометрии для нарезки.')
        mesher = BRepMesh_IncrementalMesh(shape, 0.05, False, 0.1, False)
        if not mesher.IsDone():
            raise ValueError('Не удалось построить треугольную сетку STEP/STP.')
        target = Path(folder) / 'mesh.stl'
        writer = StlAPI_Writer()
        writer.ASCIIMode = False
        if not writer.Write(shape, str(target)):
            raise ValueError('Не удалось преобразовать STEP/STP в треугольную сетку.')
        return trimesh.load(str(target), force='mesh', process=True)
