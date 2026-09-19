from __future__ import annotations

import csv
import os

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkImageData  # type: ignore
from vtkmodules.util.numpy_support import numpy_to_vtk  # type: ignore


def load_csv_checkpoint_to_imagedata(file_path: str) -> vtkImageData:
    """
    Load a CSV checkpoint or field table into vtkImageData in cell-index space. The header states the dimension: `x,u,p` is 1D, `x,y,u,v,p` is 2D and `x,y,z,u,v,w,p` is 3D. A case below 3D becomes a grid one point deep on each axis it does not have, and the velocity components it does not have stay zero.

    A line that starts with `#` before the header states where the run had reached, as `# step=400 time=0.8 dt=0.002`. Those three values reach the field data of the image, under the names a `.vtr` file uses for them, and a file written without the line loads with no field data.

    The spacing is 1.0 on every axis, so the result is indexed in cells and not in meters. A checkpoint's embedded case carries its physical extent, but this loader keeps the field table in index space.
    """

    absolute_path = os.path.abspath(file_path)
    stamp: dict[str, float] = {}
    with open(absolute_path, newline="") as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader, None)
        while header is not None and header and header[0].lstrip().startswith("#"):
            if header[0].lstrip().startswith("# step="):
                for field in header[0].lstrip("# ").split():
                    name, _, value = field.partition("=")
                    if value:
                        stamp[name] = float(value)
            header = next(csv_reader, None)
        if header is None:
            raise ValueError(f"{absolute_path} is empty; expected a header row such as x,y,z,u,v,w,p.")
        names = [cell.strip() for cell in header if cell.strip()]
        dimension = (len(names) - 1) // 2
        if dimension not in (1, 2, 3) or len(names) != 2 * dimension + 1:
            raise ValueError(f"{absolute_path} has the header {names!r}; expected x,u,p or x,y,u,v,p or x,y,z,u,v,w,p.")
        width = 2 * dimension + 1
        rows = []
        for data_row in csv_reader:
            cells = [cell for cell in data_row if cell.strip() != ""]
            if len(cells) < width:
                continue
            rows.append(cells[:width])

    if not rows:
        raise ValueError(f"No parsable data rows in {absolute_path}; expected rows of {','.join(names)}.")

    indices = np.array([[int(cell) for cell in row[:dimension]] for row in rows])  # (N, dimension)
    values = np.array([[float(cell) for cell in row[dimension:]] for row in rows])  # (N, dimension + 1)
    shape = tuple(int(indices[:, axis].max()) + 1 for axis in range(dimension))
    padded = shape + (1,) * (3 - dimension)
    scatter = tuple(indices[:, axis] for axis in range(dimension))  # (N, dimension) -> dimension x (N,)

    components = []
    for axis in range(3):
        component = np.zeros(shape, dtype=np.float32)
        if axis < dimension:
            component[scatter] = values[:, axis]  # (N, dimension + 1) -> (N,)
        components.append(component.reshape(padded))  # (*shape,) -> (nx, ny, nz)
    pressure = np.zeros(shape, dtype=np.float32)
    pressure[scatter] = values[:, dimension]  # (N, dimension + 1) -> (N,)
    pressure = pressure.reshape(padded)  # (*shape,) -> (nx, ny, nz)

    vtk_image = vtkImageData()
    vtk_image.SetDimensions(*padded)
    vtk_image.SetSpacing(1.0, 1.0, 1.0)

    point_data = vtk_image.GetPointData()
    # VTK expects x-fastest ordering; flatten in Fortran order
    for name, component in zip(("u", "v", "w"), components):
        vtk_component = numpy_to_vtk(component.flatten(order="F"), deep=True)  # (nx, ny, nz) -> (nx * ny * nz,)
        vtk_component.SetName(name)
        point_data.AddArray(vtk_component)
    vtk_pressure = numpy_to_vtk(pressure.flatten(order="F"), deep=True)  # (nx, ny, nz) -> (nx * ny * nz,)
    vtk_pressure.SetName("pressure")
    point_data.AddArray(vtk_pressure)
    point_data.SetActiveScalars("pressure")

    velocity = np.stack(components, axis=-1)  # 3 x (nx, ny, nz) -> (nx, ny, nz, 3)
    vtk_velocity = numpy_to_vtk(velocity.reshape(-1, 3, order="F"), deep=True)  # (nx, ny, nz, 3) -> (nx * ny * nz, 3)
    vtk_velocity.SetName("velocity")
    point_data.AddArray(vtk_velocity)
    point_data.SetActiveVectors("velocity")

    field_data = vtk_image.GetFieldData()
    for array_name, stamp_name in (("StepIndex", "step"), ("TimeValue", "time"), ("StepSize", "dt")):
        if stamp_name not in stamp:
            continue
        vtk_stamp = numpy_to_vtk(np.array([stamp[stamp_name]]), deep=True)
        vtk_stamp.SetName(array_name)
        field_data.AddArray(vtk_stamp)

    return vtk_image
