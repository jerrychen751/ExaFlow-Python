"""
Run a fixed case and write it to the directory named on the command line. Used by the rank-count independence test, which runs this under mpiexec at several rank counts, and by the restart test, which stops the same case half way with `half` and continues it with `finish`.
"""

from __future__ import annotations

import os
import sys

import mpi4py.MPI as mpi

from exaflow.config import (
    Boundaries, BoundaryCondition, Case, FaceCondition, Fluid, Grid,
    InitialConditions, OutputControl, TimeControl, UniformValue,
)
from exaflow.run import resume_case
from exaflow.session import SimulationSession

case = Case(
    fluid=Fluid(1.225, 0.3),
    grid=Grid((16, 16, 16), (1.0, 1.0, 1.0), 1),
    time=TimeControl(num_steps=12, cfl=0.25, integration_order=3),
    boundaries=Boundaries(left=FaceCondition(BoundaryCondition.INFLOW, (2.0, 1.0, 1.0))),
    initial=InitialConditions(velocity=tuple((UniformValue(1.0),) for _ in range(3))),
    outputs=OutputControl(checkpoint_frequency=6),
)
comm = mpi.COMM_WORLD
directory = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "all"

if mode == "finish":
    resume_case(os.path.join(directory, "Checkpoint_6.csv"), comm, output_directory=directory)
else:
    session = SimulationSession(case, comm, output_directory=directory)
    if mode == "half":
        for _ in range(6):
            session.advance_one_step()
    else:
        session.run_until_complete(write_initial=False)
