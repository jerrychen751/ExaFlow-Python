"""
The restart format. One checkpoint holds the whole domain plus the position of the run that wrote it, so a later process can continue from it at any rank count.
"""

from __future__ import annotations

import io
import os
import warnings
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..config.case import Case
from ..config.case_xml import parse_case
from ..fields import FlowState, TimeLevel, allocate_state
from ..mpi.gather import scatter_global_array
from ..mpi.subdomain import Subdomain
from .csv import format_field_csv, write_text_atomically
from .rectilinear import build_rectilinear_grid

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm

CHECKPOINT_FORMAT_VERSION = 1
CSV_FORMAT_PREFIX = "# exaflow_checkpoint_format="
CASE_XML_BEGIN = "# case_xml_begin"
CASE_XML_END = "# case_xml_end"


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """
    One run position and the whole domain that goes with it. `velocity` has shape (dimension, *grid.shape) and `pressure` has shape grid.shape, both float64 and both with the ghost layers stripped. `case_xml` is the text `write_case` produced for the running case, so the file needs no second file to be read.

    This is a whole-domain record. Rank 0 builds it and reads it; the other ranks never hold one.
    """

    velocity: np.ndarray
    pressure: np.ndarray
    level: TimeLevel
    case_xml: str


def write_checkpoint(path: str, checkpoint: Checkpoint) -> None:
    """
    Write one complete CSV or binary VTK checkpoint according to the path suffix. Rank 0 calls this and no other rank does. A sibling partial file is flushed and renamed over the target so a reader never opens half a checkpoint.
    """

    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        stamp, _, fields = format_field_csv(
            checkpoint.velocity,
            checkpoint.pressure,
            (0,) * checkpoint.pressure.ndim,
            checkpoint.level,
        ).partition("\n")
        case_lines = "".join(f"# {line}\n" for line in checkpoint.case_xml.splitlines())
        contents = (
            f"{CSV_FORMAT_PREFIX}{CHECKPOINT_FORMAT_VERSION}\n"
            f"{stamp}\n"
            f"{CASE_XML_BEGIN}\n"
            f"{case_lines}"
            f"{CASE_XML_END}\n"
            f"{fields}"
        )
        write_text_atomically(path, contents)
        return
    if suffix != ".vtr":
        raise ValueError(f"Checkpoint path must end in .csv or .vtr, got {path!r}.")

    case = parse_case(ElementTree.fromstring(checkpoint.case_xml))
    dataset = build_rectilinear_grid(case.grid, list(checkpoint.velocity), checkpoint.pressure)
    dataset.field_data["FormatVersion"] = np.array([CHECKPOINT_FORMAT_VERSION], dtype=np.int64)
    dataset.field_data["StepIndex"] = np.array([checkpoint.level.step_index], dtype=np.int64)
    dataset.field_data["TimeValue"] = np.array([checkpoint.level.current_time])
    dataset.field_data["StepSize"] = np.array([checkpoint.level.dt])
    dataset.field_data["CaseXml"] = np.array([checkpoint.case_xml])

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    root, _ = os.path.splitext(path)
    partial_path = f"{root}.partial.vtr"
    dataset.save(partial_path, binary=True)
    with open(partial_path, "rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(partial_path, path)


def read_checkpoint(path: str) -> Checkpoint:
    """
    Read one whole CSV or VTK checkpoint on one rank. A file from another format version is refused before its fields are used.
    """

    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return _read_csv_checkpoint(path)
    if suffix == ".vtr":
        return _read_vtk_checkpoint(path)
    raise ValueError(f"Checkpoint path must end in .csv or .vtr, got {path!r}.")


def _check_version(path: str, version: int) -> None:
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"{path} is checkpoint format version {version}, and this build reads version "
            f"{CHECKPOINT_FORMAT_VERSION}."
        )


def _read_csv_checkpoint(path: str) -> Checkpoint:
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    if not lines or not lines[0].startswith(CSV_FORMAT_PREFIX):
        raise ValueError(f"{path} is not an ExaFlow checkpoint: it has no format version.")
    _check_version(path, int(lines[0].removeprefix(CSV_FORMAT_PREFIX)))
    try:
        begin = lines.index(CASE_XML_BEGIN)
        end = lines.index(CASE_XML_END, begin + 1)
    except ValueError as error:
        raise ValueError(f"{path} is not an ExaFlow checkpoint: it has no embedded case XML.") from error
    if begin < 2 or end + 1 >= len(lines):
        raise ValueError(f"{path} is not an ExaFlow checkpoint: its metadata or field table is incomplete.")

    values = {}
    for field in lines[1].removeprefix("# ").split():
        name, separator, value = field.partition("=")
        if separator:
            values[name] = value
    missing = [name for name in ("step", "time", "dt") if name not in values]
    if missing:
        raise ValueError(f"{path} is not an ExaFlow checkpoint: its run position is missing {', '.join(missing)}.")
    level = TimeLevel(int(values["step"]), float(values["time"]), float(values["dt"]))
    case_lines = [line[2:] if line.startswith("# ") else line.removeprefix("#") for line in lines[begin + 1 : end]]
    case_xml = "\n".join(case_lines) + "\n"
    case = parse_case(ElementTree.fromstring(case_xml))
    names = [name.strip() for name in lines[end + 1].split(",")]
    expected_names = [*("x", "y", "z")[: case.dimension], *("u", "v", "w")[: case.dimension], "p"]
    if names != expected_names:
        raise ValueError(f"{path} has the field header {names!r}; expected {expected_names!r}.")

    data = np.loadtxt(io.StringIO("\n".join(lines[end + 2 :])), delimiter=",", ndmin=2)
    expected_rows = int(np.prod(case.grid.shape))
    expected_columns = 2 * case.dimension + 1
    if data.shape != (expected_rows, expected_columns):
        raise ValueError(
            f"{path} has a field table of shape {data.shape}; expected {(expected_rows, expected_columns)}."
        )
    index_values = data[:, : case.dimension]
    if not np.all(np.isfinite(index_values)) or not np.array_equal(index_values, np.floor(index_values)):
        raise ValueError(f"{path} has a cell index that is not an integer.")
    indices = index_values.astype(np.int64)
    if any(np.any(indices[:, axis] < 0) or np.any(indices[:, axis] >= count) for axis, count in enumerate(case.grid.shape)):
        raise ValueError(f"{path} has a cell index outside the case grid {case.grid.shape}.")
    flat_indices = np.ravel_multi_index(tuple(indices[:, axis] for axis in range(case.dimension)), case.grid.shape)
    if np.unique(flat_indices).size != expected_rows:
        raise ValueError(f"{path} does not contain every case-grid cell exactly once.")

    velocity = np.empty((case.dimension, *case.grid.shape))
    pressure = np.empty(case.grid.shape)
    velocity.reshape(case.dimension, -1)[:, flat_indices] = data[:, case.dimension : 2 * case.dimension].T
    pressure.reshape(-1)[flat_indices] = data[:, 2 * case.dimension]
    return Checkpoint(velocity, pressure, level, case_xml)


def _read_vtk_checkpoint(path: str) -> Checkpoint:
    import pyvista as pv

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated",
            category=DeprecationWarning,
        )
        dataset = pv.read(path)
        required_fields = ("FormatVersion", "StepIndex", "TimeValue", "StepSize", "CaseXml")
        missing_fields = [name for name in required_fields if name not in dataset.field_data]
        if missing_fields:
            raise ValueError(
                f"{path} is not an ExaFlow checkpoint: its field data is missing {', '.join(missing_fields)}."
            )
        _check_version(path, int(dataset.field_data["FormatVersion"][0]))
        case_xml = str(dataset.field_data["CaseXml"][0])
        case = parse_case(ElementTree.fromstring(case_xml))
        required_arrays = ("velocity", "pressure")
        missing_arrays = [name for name in required_arrays if name not in dataset.point_data]
        if missing_arrays:
            raise ValueError(
                f"{path} is not an ExaFlow checkpoint: its point data is missing {', '.join(missing_arrays)}."
            )
        expected_points = int(np.prod(case.grid.shape))
        velocity_values = np.asarray(dataset.point_data["velocity"])
        pressure_values = np.asarray(dataset.point_data["pressure"])
        level = TimeLevel(
            int(dataset.field_data["StepIndex"][0]),
            float(dataset.field_data["TimeValue"][0]),
            float(dataset.field_data["StepSize"][0]),
        )
    if velocity_values.shape != (expected_points, 3) or pressure_values.shape != (expected_points,):
        raise ValueError(
            f"{path} fields have shapes {velocity_values.shape} and {pressure_values.shape}; expected "
            f"{(expected_points, 3)} and {(expected_points,)}."
        )
    velocity = np.stack(
        [velocity_values[:, axis].reshape(case.grid.shape, order="F") for axis in range(case.dimension)]
    )
    pressure = pressure_values.reshape(case.grid.shape, order="F")
    return Checkpoint(velocity, pressure, level, case_xml)


def read_checkpoint_for_resume(
    path: str,
    comm: Intracomm | None = None,
) -> tuple[str, Checkpoint | None]:
    is_parallel = comm is not None and int(comm.Get_size()) > 1
    checkpoint = None
    text = ""
    failure = ""
    if comm is None or int(comm.Get_rank()) == 0:
        try:
            checkpoint = read_checkpoint(path)
            text = checkpoint.case_xml
        except Exception as error:
            if not is_parallel:
                raise
            failure = f"{type(error).__name__}: {error}"
    if is_parallel:
        assert comm is not None
        text, failure = comm.bcast((text, failure), root=0)
        if failure:
            raise ValueError(f"rank 0 could not read {path}: {failure}")
    return text, checkpoint


def read_case_text(path: str, comm: Intracomm | None = None) -> str:
    """
    The input XML text a checkpoint carries, on every rank. Rank 0 reads the checkpoint and broadcasts its case text.

    This is a collective call: every rank must reach it. Raises ValueError when the file holds no case text. A rank 0 that cannot read the file at all broadcasts the failure and every rank raises it, because a raise on rank 0 alone would leave the others waiting at the broadcast forever.
    """

    text, _ = read_checkpoint_for_resume(path, comm)
    return text


def scatter_checkpoint(
    path: str,
    case: Case,
    subdomain: Subdomain,
    comm: Intracomm | None = None,
    *,
    checkpoint: Checkpoint | None = None,
) -> tuple[FlowState, TimeLevel]:
    """
    Spread one checkpoint over the ranks. Rank 0 reads the file unless the caller already loaded it, and every rank gets the block of the domain its subdomain owns, plus the level rank 0 broadcasts.

    This is a collective call: every rank must reach it. The returned state holds the stored interior values and zero ghost layers, so the caller writes the prescribed boundary values before the first step. Raises ValueError when the stored domain has a different shape from the case, because the blocks would not fit. A rank 0 that cannot read the file broadcasts the failure and every rank raises it, because a raise on rank 0 alone would leave the others waiting at the broadcast forever.
    """

    is_parallel = comm is not None and int(comm.Get_size()) > 1
    failure = ""
    if comm is None or int(comm.Get_rank()) == 0:
        try:
            if checkpoint is None:
                checkpoint = read_checkpoint(path)
            if checkpoint.pressure.shape != tuple(case.grid.shape):
                raise ValueError(
                    f"{path} holds a domain of shape {checkpoint.pressure.shape}, and the case asks for "
                    f"{tuple(case.grid.shape)}."
                )
        except Exception as error:
            if not is_parallel:
                raise
            checkpoint = None
            failure = f"{type(error).__name__}: {error}"

    level = checkpoint.level if checkpoint is not None else None
    if is_parallel:
        assert comm is not None
        level, failure = comm.bcast((level, failure), root=0)
        if failure:
            raise ValueError(f"rank 0 could not read {path}: {failure}")
    if level is None:
        raise ValueError(f"{path} gave no run position to continue from.")

    state = allocate_state(subdomain, case.dimension)
    for axis in range(case.dimension):
        component = None if checkpoint is None else checkpoint.velocity[axis]  # (dimension, *shape) -> (*shape,)
        state.velocity[axis][subdomain.interior] = scatter_global_array(subdomain, comm, component)  # (dimension, *padded_shape) -> (*shape,)
    pressure = None if checkpoint is None else checkpoint.pressure
    state.pressure[subdomain.interior] = scatter_global_array(subdomain, comm, pressure)
    return state, level
