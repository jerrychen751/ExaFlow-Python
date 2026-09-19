from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from exaflow.config import Case, Fluid, Grid, InitialConditions, OutputControl, TimeControl, UniformValue
from exaflow.gui.csv_loader import load_csv_checkpoint_to_imagedata
from exaflow.session import SimulationSession

pytestmark = pytest.mark.gui


def write_csv_checkpoint(directory: Path, shape: tuple[int, ...]) -> str:
    """
    Run one step of a case of this shape, write a CSV checkpoint, and return its path. Each velocity component starts at its axis number plus one, so a component that lands on the wrong axis shows up as the wrong value.
    """

    case = Case(
        fluid=Fluid(1.225, 0.3),
        grid=Grid(shape, tuple(1.0 for _ in shape), 1),
        time=TimeControl(1, 0.25, 1),
        initial=InitialConditions(
            velocity=tuple((UniformValue(float(axis + 1)),) for axis in range(len(shape))),
            pressure=(UniformValue(7.5),),
        ),
        outputs=OutputControl(checkpoint_frequency=1),
    )
    SimulationSession(case, output_directory=str(directory)).run_until_complete()
    return str(directory / "Checkpoint_Final.csv")


@pytest.mark.parametrize("shape", [(8,), (8, 6), (8, 6, 4)])
def test_the_loader_reads_every_dimension(tmp_path: Path, shape: tuple[int, ...]) -> None:
    image = load_csv_checkpoint_to_imagedata(write_csv_checkpoint(tmp_path, shape))

    assert image.GetDimensions() == (*shape, *(1,) * (3 - len(shape)))
    assert image.GetSpacing() == (1.0, 1.0, 1.0)
    point_data = image.GetPointData()
    names = [point_data.GetArrayName(index) for index in range(point_data.GetNumberOfArrays())]
    assert names == ["u", "v", "w", "pressure", "velocity"]


def test_a_component_the_case_does_not_have_stays_zero(tmp_path: Path) -> None:
    image = load_csv_checkpoint_to_imagedata(write_csv_checkpoint(tmp_path, (8, 6)))

    point_data = image.GetPointData()
    assert np.allclose(np.array(point_data.GetArray("pressure")), 7.5)
    assert np.allclose(np.array(point_data.GetArray("w")), 0.0)
    assert np.array(point_data.GetArray("u")).max() == pytest.approx(1.0)
    assert np.array(point_data.GetArray("v")).max() == pytest.approx(2.0)


def test_the_active_arrays_are_the_ones_the_viewer_draws(tmp_path: Path) -> None:
    image = load_csv_checkpoint_to_imagedata(write_csv_checkpoint(tmp_path, (8, 6)))

    point_data = image.GetPointData()
    assert point_data.GetScalars().GetName() == "pressure"
    assert point_data.GetVectors().GetName() == "velocity"
    assert point_data.GetArray("velocity").GetNumberOfComponents() == 3


def test_a_cell_keeps_its_index_after_the_round_trip(tmp_path: Path) -> None:
    """
    The CSV runs with the last axis varying fastest and VTK expects the first axis to vary fastest, so the loader flattens in Fortran order. A wrong order swaps the axes of the picture and stays invisible on a cube.
    """

    path = tmp_path / "Checkpoint_Final.csv"
    path.write_text("x,y,u,v,p\n0, 0, 0.0, 0.0, 1.0\n0, 1, 0.0, 0.0, 2.0\n1, 0, 0.0, 0.0, 3.0\n1, 1, 0.0, 0.0, 4.0\n")

    image = load_csv_checkpoint_to_imagedata(str(path))

    assert image.GetDimensions() == (2, 2, 1)
    assert np.array(image.GetPointData().GetArray("pressure")).tolist() == [1.0, 3.0, 2.0, 4.0]


def test_a_row_shorter_than_the_header_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "Checkpoint_Ragged.csv"
    path.write_text("x,u,p\n0, 1.0, 2.0\n1\n1, 3.0, 4.0\n")

    image = load_csv_checkpoint_to_imagedata(str(path))

    assert image.GetDimensions() == (2, 1, 1)
    assert np.array(image.GetPointData().GetArray("u")).tolist() == [1.0, 3.0]


def test_a_header_that_is_not_a_field_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "Checkpoint_Wrong.csv"
    path.write_text("time,step\n0.0,1\n")

    with pytest.raises(ValueError, match="expected x,u,p"):
        load_csv_checkpoint_to_imagedata(str(path))


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "Checkpoint_Empty.csv"
    path.write_text("")

    with pytest.raises(ValueError, match="is empty"):
        load_csv_checkpoint_to_imagedata(str(path))


def test_a_file_with_a_header_and_no_rows_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "Checkpoint_Headed.csv"
    path.write_text("x,u,p\n")

    with pytest.raises(ValueError, match="No parsable data rows"):
        load_csv_checkpoint_to_imagedata(str(path))
