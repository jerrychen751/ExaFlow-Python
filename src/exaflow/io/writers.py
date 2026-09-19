from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..config.case import Case
from ..config.case_xml import write_case
from ..config.time_control import CheckpointFormat
from ..fields import FlowState, TimeLevel
from ..mpi.gather import gather_global_array
from ..mpi.subdomain import Subdomain
from .checkpoint import Checkpoint, write_checkpoint
from .rectilinear import build_rectilinear_grid

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


class WholeDomainWriter(ABC):
    def __init__(
        self,
        case: Case,
        subdomain: Subdomain,
        comm: Intracomm | None,
        *,
        frequency: int = -1,
    ) -> None:
        self.frequency = frequency
        self._case = case
        self._subdomain = subdomain
        self._comm = comm

    def write(self, label: str, state: FlowState, level: TimeLevel) -> None:
        assembled = gather_domain_fields(self._subdomain, self._comm, state)
        if assembled is None:
            return
        self._write_domain(label, *assembled, level)

    @abstractmethod
    def _write_domain(
        self,
        label: str,
        components: list[np.ndarray],
        pressure: np.ndarray,
        level: TimeLevel,
    ) -> None:
        raise NotImplementedError


class CheckpointWriter(WholeDomainWriter):
    def __init__(
        self,
        directory: str,
        case: Case,
        subdomain: Subdomain,
        comm: Intracomm | None,
        *,
        frequency: int = -1,
    ) -> None:
        super().__init__(case, subdomain, comm, frequency=frequency)
        self._directory = directory
        self._case_xml = write_case(case)

    def _write_domain(
        self,
        label: str,
        components: list[np.ndarray],
        pressure: np.ndarray,
        level: TimeLevel,
    ) -> None:
        checkpoint = Checkpoint(np.stack(components), pressure, level, self._case_xml)
        write_checkpoint(os.path.join(self._directory, f"Checkpoint_{label}{self._get_extension()}"), checkpoint)

    @abstractmethod
    def _get_extension(self) -> str:
        raise NotImplementedError


class CsvCheckpointWriter(CheckpointWriter):
    def _get_extension(self) -> str:
        return ".csv"


class VtkCheckpointWriter(CheckpointWriter):
    def _get_extension(self) -> str:
        return ".vtr"


def build_checkpoint_writer(
    directory: str,
    case: Case,
    subdomain: Subdomain,
    comm: Intracomm | None,
) -> CheckpointWriter:
    writer_type = VtkCheckpointWriter if case.outputs.checkpoint_format is CheckpointFormat.VTK else CsvCheckpointWriter
    return writer_type(
        directory,
        case,
        subdomain,
        comm,
        frequency=case.outputs.checkpoint_frequency,
    )


class StreamWriter(WholeDomainWriter):
    def __init__(
        self,
        port: int,
        case: Case,
        subdomain: Subdomain,
        comm: Intracomm | None,
        *,
        frequency: int = -1,
    ) -> None:
        super().__init__(case, subdomain, comm, frequency=frequency)
        self._port = port

    def _write_domain(
        self,
        label: str,
        components: list[np.ndarray],
        pressure: np.ndarray,
        level: TimeLevel,
    ) -> None:
        dataset = build_rectilinear_grid(self._case.grid, components, pressure)

        from ..gui.streaming.client import StreamingClient

        dataset.field_data["TimeValue"] = np.array([level.current_time])
        dataset.field_data["StepIndex"] = np.array([level.step_index])
        dataset.field_data["StepSize"] = np.array([level.dt])
        StreamingClient(port=self._port).send_dataset(dataset)
