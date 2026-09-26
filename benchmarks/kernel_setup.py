from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exaflow.config import BoundaryCondition, Boundaries, Case, FaceCondition, Fluid, Grid, TimeControl
from exaflow.fields import FlowState, allocate_state
from exaflow.mpi.process_grid import ProcessGrid
from exaflow.mpi.subdomain import Subdomain
from exaflow.numerics.time_step import compute_time_step


@dataclass(frozen=True, slots=True)
class KernelInputs:
    case: Case
    subdomain: Subdomain
    state: FlowState
    rate: FlowState
    dt: float


def build_inputs(n: int, *, integration_order: int = 1, seed: int = 0) -> KernelInputs:
    periodic = FaceCondition(BoundaryCondition.PERIODIC)
    case = Case(
        fluid=Fluid(rho=1.0, nu=0.01),
        grid=Grid((n, n, n), (1.0, 1.0, 1.0), 1),
        time=TimeControl(1, 0.5, integration_order),
        boundaries=Boundaries(left=periodic, right=periodic),
    )
    subdomain = Subdomain(case.grid, ProcessGrid((1, 1, 1)), rank=0)
    rng = np.random.default_rng(seed)
    state = allocate_state(subdomain, 3)
    state.velocity[...] = rng.uniform(-1.0, 1.0, state.velocity.shape)
    state.pressure[...] = rng.uniform(-1.0, 1.0, state.pressure.shape)
    dt = compute_time_step(case, state.compute_max_speed())
    return KernelInputs(case, subdomain, state, allocate_state(subdomain, 3), dt)
