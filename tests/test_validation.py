from __future__ import annotations

import math

import numpy as np
import pytest

from exaflow.config import Boundaries, BoundaryCondition, Case, FaceCondition, Fluid, Grid, SolverOptions, TimeControl
from exaflow.session import SimulationSession

PERIODIC = FaceCondition(BoundaryCondition.PERIODIC)


def test_the_taylor_green_vortex_converges_to_the_exact_decay() -> None:
    errors = [measure_taylor_green_error(points, include_pressure=True) for points in (16, 32, 64)]

    assert errors[2] < 0.05
    assert math.log2(errors[0] / errors[1]) > 0.8
    assert math.log2(errors[1] / errors[2]) > 0.8


def measure_taylor_green_error(points: int, include_pressure: bool) -> float:
    nu = 0.05
    spacing = 2 * math.pi / points
    case = Case(
        fluid=Fluid(1.0, nu),
        grid=Grid((points, points), (2 * math.pi, 2 * math.pi), 1),
        time=TimeControl(num_steps=1000, cfl=0.5, integration_order=3, end_time=1.0),
        boundaries=Boundaries(left=PERIODIC, right=PERIODIC, top=PERIODIC, bottom=PERIODIC),
        solver=SolverOptions(include_pressure=include_pressure),
    )
    session = SimulationSession(case)
    coordinate = (np.arange(points + 2) - 0.5) * spacing
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    session.state.velocity[0][...] = np.sin(x) * np.cos(y)
    session.state.velocity[1][...] = -np.cos(x) * np.sin(y)
    session.dt = session.choose_time_step()
    session.run_until_complete(write_initial=False)

    decay = math.exp(-2 * nu * session.current_time)
    interior = session.subdomain.interior
    return max(
        float(np.abs(session.state.velocity[0][interior] - np.sin(x[interior]) * np.cos(y[interior]) * decay).max()),
        float(np.abs(session.state.velocity[1][interior] + np.cos(x[interior]) * np.sin(y[interior]) * decay).max()),
    )


def test_the_taylor_green_vortex_needs_the_pressure_projection() -> None:
    without = measure_taylor_green_error(64, include_pressure=False)
    with_pressure = measure_taylor_green_error(64, include_pressure=True)

    assert without > 5 * with_pressure


def test_a_pressure_driven_channel_approaches_the_poiseuille_profile() -> None:
    errors = []
    for points in (9, 17):
        velocity, exact = run_pressure_driven_channel(points)
        assert np.abs(velocity[1]).max() < 1e-12
        errors.append(float(np.abs(velocity[0] - exact).max()))

    assert math.log2(errors[0] / errors[1]) > 1.7


def run_pressure_driven_channel(points: int) -> tuple[np.ndarray, np.ndarray]:
    case = Case(
        fluid=Fluid(1.0, 1.0),
        grid=Grid((2 * points - 1, points), (2.0, 1.0), 1),
        time=TimeControl(num_steps=100000, cfl=0.5, end_time=2.0),
        boundaries=Boundaries(
            left=FaceCondition(BoundaryCondition.OUTFLOW, pressure=1.0),
            right=FaceCondition(BoundaryCondition.OUTFLOW, pressure=0.0),
        ),
        solver=SolverOptions(include_pressure=True),
    )
    session = SimulationSession(case)
    session.run_until_complete(write_initial=False)

    height = (np.arange(points) + 0.5) * case.grid.spacing[1]
    return session.state.velocity[(slice(None), *session.subdomain.interior)], 0.25 * height * (1.0 - height)


def test_a_wall_sits_on_the_domain_face() -> None:
    velocity, exact = run_pressure_driven_channel(17)

    assert np.abs(velocity[0] - exact).max() < 0.01 * exact.max()
