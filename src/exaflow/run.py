from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from typing import TYPE_CHECKING

from .config import Case
from .config.case_xml import parse_case
from .fields import FlowState
from .io.checkpoint import read_checkpoint_for_resume
from .session import SimulationSession

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm


def run_case(
    case: Case,
    comm: Intracomm | None = None,
    *,
    output_directory: str | None = None,
    stream_port: int | None = None,
) -> FlowState:
    """
    Run one typed Case on this rank and return its final local state. Every MPI rank must call this function with the same Case and output directory.
    """

    return SimulationSession(case, comm, output_directory=output_directory, stream_port=stream_port).run_until_complete()


def resume_case(
    checkpoint_path: str,
    comm: Intracomm | None = None,
    *,
    case: Case | None = None,
    output_directory: str | None = None,
    stream_port: int | None = None,
) -> FlowState:
    """
    Continue the run one checkpoint holds, and return its final local state. The case comes from the checkpoint unless `case` replaces it, which is how a finished run is extended with a larger `num_steps` or a later `end_time`. A replacement case must describe the same grid shape, because the stored arrays fit no other one.

    Every MPI rank must call this function with the same arguments.
    """

    stored_case_text, checkpoint = read_checkpoint_for_resume(checkpoint_path, comm)
    stored_case = parse_case(ElementTree.fromstring(stored_case_text))
    if case is None:
        case = stored_case
    elif case.grid.shape != stored_case.grid.shape:
        raise ValueError(
            f"{checkpoint_path} holds a domain of shape {stored_case.grid.shape}, and the case given asks "
            f"for {case.grid.shape}."
        )

    session = SimulationSession(
        case,
        comm,
        output_directory=output_directory,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        stream_port=stream_port,
    )
    return session.run_until_complete()
