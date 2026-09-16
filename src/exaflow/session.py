from __future__ import annotations

import os
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .boundary_application import initialize_boundaries
from .config.case import Case
from .config.case_xml import write_case
from .fields import FlowState, TimeLevel, build_initial_state
from .io.checkpoint import Checkpoint, scatter_checkpoint, write_checkpoint
from .io.writers import StreamWriter, Writer, build_writers, gather_domain_fields
from .mpi.process_grid import ProcessGrid, choose_process_grid
from .mpi.subdomain import Subdomain
from .numerics.operators import SpatialOperator
from .numerics.time_step import TimeIntegrator, compute_time_step

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm


class SimulationSession:
    """
    One run in progress on this rank. The session owns everything that belongs to a run rather than to the problem: the communicator, the decomposition, the fields, how far the run has gone, the stage buffers and the output schedule. The case itself stays frozen throughout.

    `state`, `step_index`, `current_time` and `dt` move together, so a caller that stops between steps can read where the run is. `advance_one_step` replaces `state`, so a reference kept across a step points at a stage buffer the next step overwrites.

    Every method is collective. Build the session on every rank with the same case, and call the same methods on every rank in the same order.
    """

    def __init__(
        self,
        case: Case,
        comm: Intracomm | None = None,
        *,
        output_directory: str | None = None,
        writers: Sequence[Writer] | None = None,
        checkpoint_path: str | None = None,
        stream_port: int | None = None,
    ) -> None:
        self.case = case
        self.comm = comm
        self.rank = int(comm.Get_rank()) if comm is not None else 0
        num_procs = int(comm.Get_size()) if comm is not None else 1

        self.process_grid = ProcessGrid(choose_process_grid(num_procs, case.grid.shape, case.grid.num_ghost_layers))
        self.subdomain = Subdomain(case.grid, self.process_grid, self.rank)
        self.output_directory = output_directory

        if writers is not None:
            self.writers: tuple[Writer, ...] = tuple(writers)
        elif output_directory is not None:
            self.writers = build_writers(output_directory, case.grid, self.subdomain, comm, case.outputs)
        else:
            self.writers = ()
        if stream_port is not None:
            self.writers += (
                StreamWriter(stream_port, case.grid, self.subdomain, comm, frequency=case.outputs.total_frequency),
            )
        if case.outputs.checkpoint_frequency != -1 and output_directory is None:
            raise ValueError("A case with a checkpoint interval needs an output directory to write into.")
        self._case_xml = write_case(case) if output_directory is not None else ""

        if checkpoint_path is None:
            self.state = self.build_initial_state()
            self.step_index = 0
            self.current_time = 0.0
            self.dt = self.choose_time_step()
        else:
            self.state, level = scatter_checkpoint(checkpoint_path, case, self.subdomain, comm)
            initialize_boundaries(self.state, case, self.subdomain)
            self.step_index = level.step_index
            self.current_time = level.current_time
            self.dt = level.dt

        self._integrator = TimeIntegrator(
            SpatialOperator(case, self.subdomain, comm),
            case.time.integration_order,
            self.state,
        )

    def build_initial_state(self) -> FlowState:
        """
        The starting fields for this rank, with the prescribed boundary values already written into the ghost layers of every domain face this rank owns.
        """

        state = build_initial_state(self.case, self.subdomain)
        initialize_boundaries(state, self.case, self.subdomain)
        return state

    def choose_time_step(self) -> float:
        """
        The largest stable step in seconds for the state this session holds now, reduced over every rank so that all ranks march together.
        """

        local_max = self.state.compute_max_speed()
        if self.comm is not None:
            from mpi4py import MPI

            local_max = float(self.comm.allreduce(local_max, op=MPI.MAX))
        return compute_time_step(self.case, local_max)

    @property
    def level(self) -> TimeLevel:
        """
        Where this run has reached, as the one record a writer and a checkpoint both take.
        """

        return TimeLevel(self.step_index, self.current_time, self.dt)

    def is_complete(self) -> bool:
        """
        Report whether the run has reached its target. `case.time.num_steps` is the step budget of the whole run counted from time zero, so a session restored at step 400 of 1000 reports False until it has taken 600 more steps. An end time stops the run as soon as `current_time` reaches it.
        """

        if self.step_index >= self.case.time.num_steps:
            return True
        end_time = self.case.time.end_time
        return end_time is not None and self.current_time >= end_time

    def write(self, label: str) -> None:
        """
        Call every writer once with this label, whatever its interval. A frequency of -1 suppresses the interval writes only; the first and last state are written through every writer.
        """

        level = self.level
        for writer in self.writers:
            writer.write(label, self.state, level)

    def save_checkpoint(self, label: str) -> None:
        """
        Write `Checkpoint_<label>.npz` into the output directory of this run. Rank 0 writes the file; the other ranks take part in the gather and write nothing.

        This is a collective call: every rank must reach it. Raises ValueError when the session was given no output directory. The case text goes into the file as `__init__` rendered it, because a render here would raise on rank 0 alone, mid-march, and leave the other ranks at the next ghost exchange.
        """

        if self.output_directory is None:
            raise ValueError("This session has no output directory, so it cannot write a checkpoint.")
        assembled = gather_domain_fields(self.subdomain, self.comm, self.state)
        if assembled is None:
            return
        components, pressure = assembled
        velocity = np.stack(components)  # dimension x (*shape,) -> (dimension, *shape)
        checkpoint = Checkpoint(velocity, pressure, self.level, self._case_xml)
        write_checkpoint(os.path.join(self.output_directory, f"Checkpoint_{label}.npz"), checkpoint)

    def advance_one_step(self) -> None:
        """
        Take one step, then write what that step selects: every writer whose interval names it, and a checkpoint when the checkpoint interval names it. The label of both is the completed step count, so the file, the level inside it and the step a restart begins at are the same number.

        The caller tests `is_complete` first, because a run that has reached its end time would otherwise take a step of zero or less. This replaces `state`, so a reference the caller kept before the call is a stale buffer.
        """

        if self.is_complete():
            raise RuntimeError(f"The run is complete at step {self.step_index} of {self.case.time.num_steps}.")

        if self.case.time.adaptive_time_step:
            self.dt = self.choose_time_step()
        end_time = self.case.time.end_time
        if end_time is not None and end_time - self.current_time <= self.dt:
            step_size = end_time - self.current_time
            next_time = end_time
        else:
            step_size = self.dt
            next_time = self.current_time + self.dt

        self.state = self._integrator.advance(self.state, step_size)
        self.step_index += 1
        self.current_time = next_time

        label = str(self.step_index)
        level = self.level
        for writer in self.writers:
            if self.case.outputs.is_due(writer.frequency, self.step_index):
                writer.write(label, self.state, level)
        if self.case.outputs.is_due(self.case.outputs.checkpoint_frequency, self.step_index):
            self.save_checkpoint(label)

    def run_until_complete(self, *, write_initial: bool = True) -> FlowState:
        """
        Advance until `is_complete` reports True, and return the final state for this rank. Every writer is called once more with the label "Final" at the end, and a case with a checkpoint interval writes `Checkpoint_Final.npz` there too. `write_initial` writes the starting state first, under the label "Original", or under "Resumed" when the session started from a checkpoint.
        """

        if write_initial:
            self.write("Original" if self.step_index == 0 else "Resumed")
        while not self.is_complete():
            self.advance_one_step()
        self.write("Final")
        if self.case.outputs.checkpoint_frequency != -1:
            self.save_checkpoint("Final")
        return self.state
