from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ..config.grid import Grid
from ..config.time_control import OutputControl, OutputFormat
from ..fields import FlowState, TimeLevel
from ..mpi.gather import gather_global_array
from ..mpi.subdomain import Subdomain
from .csv import format_field_csv, write_text_atomically

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm


def gather_domain_fields(
    subdomain: Subdomain,
    comm: Intracomm | None,
    state: FlowState,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    """
    Assemble every field on rank 0 as (velocity components in axis order, pressure), with the ghost layers stripped, or None on every other rank.

    This is a collective call: every rank must reach it, because the gather reduces across all of them.
    """

    interior = subdomain.interior
    components = [
        gather_global_array(subdomain, comm, state.velocity[axis][interior])
        for axis in range(state.dimension)
    ]
    pressure = gather_global_array(subdomain, comm, state.pressure[interior])
    if pressure is None or any(part is None for part in components):
        return None
    return [part for part in components if part is not None], pressure


class Writer(Protocol):
    """
    One output format. `frequency` is the interval in steps, or -1 to write only when the session asks directly. `level` states where the run had reached, and the file records it. `write` is collective: every rank must call it, because a writer that assembles the full domain reduces across ranks.
    """

    frequency: int

    def write(self, label: str, state: FlowState, level: TimeLevel) -> None: ...


class RankCsvWriter:
    """
    One CSV file per rank per label, named `<label>_<rank>.csv`, holding only the cells that rank owns. Writes no message between ranks, so it stays cheap at every interval.
    """

    def __init__(self, directory: str, subdomain: Subdomain, *, frequency: int = -1) -> None:
        self.frequency = frequency
        self._directory = directory
        self._subdomain = subdomain

    def write(self, label: str, state: FlowState, level: TimeLevel) -> None:
        interior = self._subdomain.interior
        velocity = np.stack([state.velocity[axis][interior] for axis in range(state.dimension)])  # dimension x (*shape,) -> (dimension, *shape)
        origin = tuple(start for start, _ in self._subdomain.bounds)
        text = format_field_csv(velocity, state.pressure[interior], origin, level)
        write_text_atomically(os.path.join(self._directory, f"{label}_{self._subdomain.rank}.csv"), text)


class TotalCsvWriter:
    """
    One CSV file per label holding the whole domain, named `<label>_Total.csv`. Rank 0 assembles the blocks and writes; the other ranks take part in the gather and write nothing.
    """

    def __init__(self, directory: str, subdomain: Subdomain, comm: Intracomm | None, *, frequency: int = -1) -> None:
        self.frequency = frequency
        self._directory = directory
        self._subdomain = subdomain
        self._comm = comm

    def write(self, label: str, state: FlowState, level: TimeLevel) -> None:
        assembled = gather_domain_fields(self._subdomain, self._comm, state)
        if assembled is None:
            return
        components, pressure = assembled
        text = format_field_csv(np.stack(components), pressure, (0,) * self._subdomain.grid.dimension, level)  # dimension x (*shape,) -> (dimension, *shape)
        write_text_atomically(os.path.join(self._directory, f"{label}_Total.csv"), text)


def pad_to_three_axes(
    grid: Grid,
    components: list[np.ndarray],
    pressure: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    axes = [
        np.linspace(0.0, float(span), int(count))
        for span, count in zip(grid.extent, grid.shape)
    ]
    while len(axes) < 3:
        axes.append(np.array([0.0], dtype=float))
    padded = pressure.shape + (1,) * (3 - pressure.ndim)
    pressure = np.ascontiguousarray(pressure.reshape(padded))  # (*shape,) -> (nx, ny, nz)
    components = [np.ascontiguousarray(part.reshape(padded)) for part in components]  # (*shape,) -> (nx, ny, nz) each
    while len(components) < 3:
        components.append(np.zeros(padded))
    return axes, components, pressure


class VtkWriter:
    """
    One VTK rectilinear grid per label holding the whole domain, named `<label>_Total.vtr`. Rank 0 assembles the blocks and writes; the other ranks take part in the gather and write nothing. Needs pyevtk, which is imported only when a write happens.

    The file states where the run had reached in the field data of the grid. `TimeValue` is the array name ParaView reads as the time of a file. pyevtk writes its own `fieldData` argument inside the `<Piece>` element, where no reader looks for it, so the block is written into the grid element here instead.
    """

    def __init__(
        self,
        directory: str,
        grid: Grid,
        subdomain: Subdomain,
        comm: Intracomm | None,
        *,
        frequency: int = -1,
    ) -> None:
        self.frequency = frequency
        self._directory = directory
        self._grid = grid
        self._subdomain = subdomain
        self._comm = comm

    def write(self, label: str, state: FlowState, level: TimeLevel) -> None:
        try:
            from pyevtk.hl import gridToVTK  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError("VTK output needs the 'pyevtk' package.") from error

        assembled = gather_domain_fields(self._subdomain, self._comm, state)
        if assembled is None:
            return
        axes, components, pressure = pad_to_three_axes(self._grid, *assembled)

        os.makedirs(self._directory, exist_ok=True)
        gridToVTK(
            os.path.join(self._directory, f"{label}_Total"),
            axes[0],
            axes[1],
            axes[2],
            pointData={"pressure": pressure, "velocity": tuple(components)},
        )

        arrays = "".join(
            f'<DataArray type="{kind}" Name="{name}" NumberOfTuples="1" format="ascii">{value!r}</DataArray>\n'
            for name, kind, value in (
                ("TimeValue", "Float64", level.current_time),
                ("StepIndex", "Int64", level.step_index),
                ("StepSize", "Float64", level.dt),
            )
        )
        written = os.path.join(self._directory, f"{label}_Total.vtr")
        with open(written, "rb") as handle:
            raw = handle.read()
        partial_path = f"{written}.partial"
        with open(partial_path, "wb") as handle:
            handle.write(raw.replace(b"<Piece", f"<FieldData>\n{arrays}</FieldData>\n<Piece".encode(), 1))
        os.replace(partial_path, written)


def build_writers(
    directory: str,
    grid: Grid,
    subdomain: Subdomain,
    comm: Intracomm | None,
    outputs: OutputControl,
) -> tuple[Writer, ...]:
    """
    The writers for one run, each carrying the interval its file was given. Every writer belongs to the one format the case selected, so a run folder holds .vtr files or .csv files and never both. An interval of -1 still gets a writer, because the session writes the first and last state through every writer whatever its interval.
    """

    if outputs.format is OutputFormat.VTK:
        return (VtkWriter(directory, grid, subdomain, comm, frequency=outputs.total_frequency),)
    return (
        TotalCsvWriter(directory, subdomain, comm, frequency=outputs.total_frequency),
        RankCsvWriter(directory, subdomain, frequency=outputs.partial_frequency),
    )
