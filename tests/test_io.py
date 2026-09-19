from __future__ import annotations

import string
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from exaflow.config import Case, CheckpointFormat, OutputControl
from exaflow.fields import TimeLevel, allocate_state
from exaflow.io.csv import format_field_csv, write_text_atomically
from exaflow.io.storage import create_run_directory, resolve_output_root
from exaflow.io.writers import CsvCheckpointWriter, VtkCheckpointWriter, build_checkpoint_writer, gather_domain_fields
from exaflow.mpi.gather import gather_global_array
from exaflow.mpi.subdomain import Subdomain

LEVEL = TimeLevel(step_index=4, current_time=0.5, dt=0.125)


@pytest.mark.parametrize(
    "dimension,expected",
    [(1, "x,u,p"), (2, "x,y,u,v,p"), (3, "x,y,z,u,v,w,p")],
)
def test_the_header_names_the_axes_then_the_components_then_pressure(
    dimension: int,
    expected: str,
) -> None:
    shape = (2,) * dimension
    text = format_field_csv(np.zeros((dimension, *shape)), np.zeros(shape), (0,) * dimension, LEVEL)
    assert text.splitlines()[:2] == ["# step=4 time=0.5 dt=0.125", expected]


def test_the_index_columns_carry_global_indices() -> None:
    """
    A rank writes its own block, so the index columns have to start at the global index of its first cell. Without the origin two ranks would write the same indices for different cells.
    """

    velocity = np.zeros((2, 2, 2))
    text = format_field_csv(velocity, np.zeros((2, 2)), (4, 7), LEVEL)
    rows = [row.split(",")[:2] for row in text.splitlines()[2:]]
    assert [[int(x), int(y)] for x, y in rows] == [[4, 7], [4, 8], [5, 7], [5, 8]]


def test_rows_run_with_the_last_axis_varying_fastest() -> None:
    velocity = np.zeros((2, 2, 2))
    pressure = np.arange(10.0, 14.0).reshape(2, 2)  # (4,) -> (2, 2)
    text = format_field_csv(velocity, pressure, (0, 0), LEVEL)
    assert [row.split(", ")[-1] for row in text.splitlines()[2:]] == ["10.0", "11.0", "12.0", "13.0"]


def test_a_value_reads_back_as_the_float_that_produced_it() -> None:
    """
    The columns are written with %r, so the text carries the shortest string that round-trips. A fixed number of decimals would lose the low bits and make a rank-count comparison fail on rounding alone.
    """

    values = np.array([0.1, 1.0 / 3.0, 1e-17, np.pi])
    text = format_field_csv(values.reshape(1, 4), values * 2.0, (0,), LEVEL)  # (4,) -> (1, 4), one component of a 1D case
    read_back = [float(row.split(", ")[1]) for row in text.splitlines()[2:]]
    assert read_back == values.tolist()


def test_an_atomic_write_creates_the_parent_and_leaves_no_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "Checkpoint_Final.csv"

    write_text_atomically(str(path), "x,u,p\n")

    assert path.read_text(encoding="utf-8") == "x,u,p\n"
    assert sorted(entry.name for entry in path.parent.iterdir()) == ["Checkpoint_Final.csv"]


def test_an_atomic_write_replaces_the_file_that_is_there(tmp_path: Path) -> None:
    path = tmp_path / "Checkpoint_Final.csv"
    write_text_atomically(str(path), "first\n")

    write_text_atomically(str(path), "second\n")

    assert path.read_text(encoding="utf-8") == "second\n"
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["Checkpoint_Final.csv"]


def test_the_output_root_follows_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "elsewhere"
    monkeypatch.setenv("EXAFLOW_OUTPUT_ROOT", str(root))

    assert resolve_output_root() == str(root)
    assert root.is_dir()


def test_the_output_root_defaults_to_the_documents_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXAFLOW_OUTPUT_ROOT", "")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert resolve_output_root() == str(tmp_path / "Documents" / "ExaFlow")


def test_a_run_directory_carries_a_timestamp_and_a_safe_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Every character outside [A-Za-z0-9-_] becomes an underscore, so a label taken from a file name cannot reach outside the output root.
    """

    monkeypatch.setenv("EXAFLOW_OUTPUT_ROOT", str(tmp_path))

    run_directory = Path(create_run_directory("../my case!"))

    assert run_directory.parent == tmp_path
    assert run_directory.is_dir()
    assert run_directory.name.endswith("my_case")
    assert set(run_directory.name) <= set(string.ascii_letters + string.digits + "-_")


def test_a_label_of_only_punctuation_leaves_the_timestamp_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXAFLOW_OUTPUT_ROOT", str(tmp_path))

    run_directory = Path(create_run_directory("///"))

    assert run_directory.is_dir()
    assert not run_directory.name.endswith("_")


def test_a_serial_gather_returns_the_block_it_was_given(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> None:
    case = build_case((4, 3))
    subdomain = build_subdomain(case.grid)
    block = np.arange(12.0).reshape(4, 3)  # (12,) -> (4, 3)

    assembled = gather_global_array(subdomain, None, block)

    assert assembled is not None
    assert np.array_equal(assembled, block)
    assert assembled.flags["C_CONTIGUOUS"]


def test_a_serial_gather_strips_the_ghost_layers(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> None:
    case = build_case((4, 3))
    subdomain = build_subdomain(case.grid)
    state = allocate_state(subdomain, 2)
    state.velocity[...] = 5.0
    state.velocity[0][subdomain.interior] = 1.0

    assembled = gather_domain_fields(subdomain, None, state)

    assert assembled is not None
    components, pressure = assembled
    assert [part.shape for part in components] == [(4, 3), (4, 3)]
    assert pressure.shape == (4, 3)
    assert np.all(components[0] == 1.0)


def test_a_csv_case_builds_a_csv_checkpoint_writer(
    tmp_path: Path,
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> None:
    case = build_case((8,))
    subdomain = build_subdomain(case.grid)

    writer = build_checkpoint_writer(str(tmp_path), case, subdomain, None)

    assert isinstance(writer, CsvCheckpointWriter)


def test_a_vtk_case_builds_a_vtk_checkpoint_writer(
    tmp_path: Path,
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> None:
    case = build_case((8,), outputs=OutputControl(checkpoint_format=CheckpointFormat.VTK, checkpoint_frequency=2))
    subdomain = build_subdomain(case.grid)

    writer = build_checkpoint_writer(str(tmp_path), case, subdomain, None)

    assert isinstance(writer, VtkCheckpointWriter)
    assert writer.frequency == 2
