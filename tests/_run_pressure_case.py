from __future__ import annotations

import sys

import mpi4py.MPI as mpi

from exaflow.config import (
    Boundaries, BoundaryCondition, Case, FaceCondition, Fluid, Grid,
    InitialConditions, OutputControl, SolverOptions, StepValue, TimeControl, UniformValue,
)
from exaflow.session import SimulationSession

periodic = FaceCondition(BoundaryCondition.PERIODIC)
case = Case(
    fluid=Fluid(1.0, 0.05),
    grid=Grid((16, 12, 10), (2.0, 1.0, 1.0), 1),
    time=TimeControl(num_steps=6, cfl=0.3, integration_order=3),
    boundaries=Boundaries(
        left=FaceCondition(BoundaryCondition.INFLOW, (1.0, 0.0, 0.0)),
        right=FaceCondition(BoundaryCondition.OUTFLOW, pressure=0.0),
        front=periodic,
        back=periodic,
    ),
    initial=InitialConditions(velocity=(
        (UniformValue(1.0), StepValue(0.5, (0.3, 0.3, 0.0), (0.6, 0.6, 1.0))),
        (UniformValue(0.0),),
        (UniformValue(0.0),),
    )),
    solver=SolverOptions(include_pressure=True),
    outputs=OutputControl(checkpoint_frequency=6),
)
SimulationSession(case, mpi.COMM_WORLD, output_directory=sys.argv[1]).run_until_complete(write_initial=False)
