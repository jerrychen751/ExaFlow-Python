"""
The time loop runs the projection solver after every Runge-Kutta stage when include_pressure is set. These tests hold the contract its docstrings state, so the term behaves as its caller was promised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from exaflow.config import (
    Boundaries, BoundaryCondition, Case, FaceCondition, Fluid, Grid,
    InitialConditions, SolverOptions, StepValue, TimeControl, UniformValue,
)
from exaflow.fields import FlowState, allocate_state
from exaflow.io.checkpoint import read_checkpoint
from exaflow.mpi.ghost_exchange import GhostExchange
from exaflow.mpi.process_grid import ProcessGrid
from exaflow.mpi.subdomain import Subdomain
from exaflow.numerics.pressure_poisson import PoissonSolver
from exaflow.session import SimulationSession

TIME_STEP = 0.25


def build_swirl(subdomain: Subdomain) -> FlowState:
    """
    A state whose velocity is smooth over the padded block and whose divergence is neither zero nor constant, so a solver that dropped a term or read the wrong axis cannot pass by accident.
    """

    padded = subdomain.padded_shape
    dimension = len(padded)
    grid = np.meshgrid(*(np.arange(length, dtype=float) for length in padded), indexing="ij")
    state = allocate_state(subdomain, dimension)
    for axis in range(dimension):
        state.velocity[axis][...] = np.sin(0.4 * grid[axis] + 0.7 * axis)
    return state


@pytest.fixture
def build_poisson_solver(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> Callable[..., tuple[PoissonSolver, Subdomain]]:
    def make(shape: tuple[int, ...], rho: float = 1.0) -> tuple[PoissonSolver, Subdomain]:
        case = build_case(shape, fluid=Fluid(rho, 0.1))
        subdomain = build_subdomain(case.grid)
        return PoissonSolver(case, subdomain), subdomain

    return make


@pytest.mark.parametrize("shape", [(8,), (8, 7), (8, 7, 6)])
def test_the_operator_is_symmetric_and_singular(
    build_poisson_solver: Callable[..., tuple[PoissonSolver, Subdomain]],
    shape: tuple[int, ...],
) -> None:
    """
    This assembles the matrix column by column from the stencil, which no caller needs and nothing else exposes. Conjugate gradient assumes a symmetric positive semi-definite matrix, and a wrong end row breaks that quietly: the run still returns an answer. A zero normal gradient on every face makes every row sum to zero, which is the same statement as the constant vector lying in the nullspace.
    """

    solver, subdomain = build_poisson_solver(shape)
    operator = assemble_operator(solver, subdomain)

    assert np.array_equal(operator, operator.T)
    assert np.allclose(operator.sum(axis=1), 0.0)
    assert np.linalg.eigvalsh(operator).min() == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.eigvalsh(operator).max() > 0.0


def assemble_operator(solver: PoissonSolver, subdomain: Subdomain) -> np.ndarray:
    size = int(np.prod(subdomain.shape))
    columns = []
    for index in range(size):
        unit = np.zeros(size)
        unit[index] = 1.0
        column = np.empty(subdomain.shape)
        solver._apply_operator(unit.reshape(subdomain.shape), column)
        columns.append(column.ravel())
    return np.column_stack(columns)


@pytest.mark.parametrize(
    "boundaries,is_singular",
    [
        (Boundaries(right=FaceCondition(BoundaryCondition.OUTFLOW, pressure=1.5)), False),
        (Boundaries(left=FaceCondition(BoundaryCondition.PERIODIC), right=FaceCondition(BoundaryCondition.PERIODIC)), True),
    ],
)
def test_outflow_and_periodic_faces_keep_the_operator_symmetric(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
    boundaries: Boundaries,
    is_singular: bool,
) -> None:
    case = build_case((6, 5, 4), boundaries=boundaries)
    subdomain = build_subdomain(case.grid)
    operator = assemble_operator(PoissonSolver(case, subdomain), subdomain)
    smallest = np.linalg.eigvalsh(operator).min()

    assert np.array_equal(operator, operator.T)
    if is_singular:
        assert smallest == pytest.approx(0.0, abs=1e-9)
    else:
        assert smallest > 1e-6


def test_still_fluid_takes_the_outflow_pressure_everywhere(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> None:
    case = build_case((8, 7, 6), boundaries=Boundaries(right=FaceCondition(BoundaryCondition.OUTFLOW, pressure=2.0)))
    subdomain = build_subdomain(case.grid)

    pressure = PoissonSolver(case, subdomain).solve(allocate_state(subdomain, 3), TIME_STEP)

    assert np.allclose(pressure, 2.0)


@pytest.mark.parametrize("shape", [(8,), (8, 7), (8, 7, 6)])
def test_a_divergence_free_field_needs_no_pressure(
    build_poisson_solver: Callable[..., tuple[PoissonSolver, Subdomain]],
    shape: tuple[int, ...],
) -> None:
    """
    A uniform field satisfies grad u = 0 everywhere, so the right-hand side is zero and the solution is zero at every interior point.
    """

    solver, subdomain = build_poisson_solver(shape)
    uniform = allocate_state(subdomain, len(shape))
    for axis, value in enumerate((3.0, -1.0, 0.5)[: len(shape)]):
        uniform.velocity[axis][...] = value

    pressure = solver.solve(uniform, TIME_STEP)

    assert pressure.shape == subdomain.shape
    assert np.allclose(pressure, 0.0)


@pytest.mark.parametrize("shape", [(10,), (10, 9), (10, 9, 8)])
def test_the_answer_has_a_mean_of_zero(
    build_poisson_solver: Callable[..., tuple[PoissonSolver, Subdomain]],
    shape: tuple[int, ...],
) -> None:
    """
    A zero normal gradient on every face fixes the pressure only up to a constant, so the solver has to choose one. It returns the zero-mean answer, which keeps a long run from drifting away from the range a float64 resolves.
    """

    solver, subdomain = build_poisson_solver(shape)

    pressure = solver.solve(build_swirl(subdomain), TIME_STEP)

    assert pressure.mean() == pytest.approx(0.0, abs=1e-12)


def test_the_pressure_scales_with_the_density(
    build_poisson_solver: Callable[..., tuple[PoissonSolver, Subdomain]],
) -> None:
    """
    The right-hand side is (rho / dt) * div u*, and the operator does not depend on rho, so three times the density gives three times the pressure for the same field.
    """

    light, subdomain = build_poisson_solver((10, 9), 1.0)
    heavy, _ = build_poisson_solver((10, 9), 3.0)
    field = build_swirl(subdomain)

    assert np.allclose(heavy.solve(field, TIME_STEP), 3.0 * light.solve(field, TIME_STEP))


def test_the_same_field_solves_to_the_same_pressure_twice(
    build_poisson_solver: Callable[..., tuple[PoissonSolver, Subdomain]],
) -> None:
    """
    Each solve keeps its answer and starts the next conjugate-gradient run from it. A warm start must change the work and not the answer.
    """

    solver, subdomain = build_poisson_solver((10, 9))
    field = build_swirl(subdomain)

    first = solver.solve(field, TIME_STEP)
    second = solver.solve(field, TIME_STEP)

    assert np.array_equal(first, second)


@pytest.mark.parametrize("shape", [(20,), (12, 11), (10, 9, 8)])
def test_the_projection_never_raises_the_divergence(
    build_poisson_solver: Callable[..., tuple[PoissonSolver, Subdomain]],
    shape: tuple[int, ...],
) -> None:
    """
    The operator this replaced ended one end row at -2 rather than -1, which states a pressure of zero one point outside the block instead of a zero normal gradient. The projection then pushed the divergence up rather than down, and further up on a finer grid.
    """

    solver, subdomain = build_poisson_solver(shape)
    state = build_swirl(subdomain)

    before = float(np.abs(solver.compute_divergence(state)).max())
    solver.project(state, TIME_STEP)
    after = float(np.abs(solver.compute_divergence(state)).max())

    assert after < before


@pytest.mark.parametrize("shape", [(20,), (12, 11), (10, 9, 8)])
def test_project_records_the_pressure_that_did_the_correction(
    build_poisson_solver: Callable[..., tuple[PoissonSolver, Subdomain]],
    shape: tuple[int, ...],
) -> None:
    """
    project is the whole corrector half of Chorin's method, so the caller keeps the pressure it used. It reaches the interior of the state and leaves the ghost layers for the exchange to fill.
    """

    solver, subdomain = build_poisson_solver(shape)
    state = build_swirl(subdomain)

    expected = solver.solve(state, TIME_STEP)
    solver.project(state, TIME_STEP)

    assert np.allclose(state.pressure[subdomain.interior], expected)
    assert state.pressure[(0,) * len(shape)] == 0.0


def test_the_correction_reaches_every_real_point(
    build_poisson_solver: Callable[..., tuple[PoissonSolver, Subdomain]],
) -> None:
    """
    The pressure is padded with the edge value before the gradient is taken, which is the zero normal gradient the operator carries. Every real point therefore gets a correction, and the ghost layers keep the predicted velocity for the caller to refresh.
    """

    solver, subdomain = build_poisson_solver((10, 9))
    state = build_swirl(subdomain)
    before = state.velocity[0].copy()

    solver.correct_velocity(state, solver.solve(state, TIME_STEP), TIME_STEP)

    changed = state.velocity[0] != before
    assert changed[subdomain.interior].all()
    assert not changed[0].any()
    assert not changed[-1].any()


@pytest.mark.parametrize("num_ghost_layers", [1, 2, 3])
def test_the_pad_depth_changes_nothing_the_solver_returns(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
    num_ghost_layers: int,
) -> None:
    """
    compute_divergence indexes inward from each end of the padded array, and the high-end bound needs a None where a single ghost layer would otherwise give a slice of zero length. The answer covers the real points whatever the pad depth.
    """

    case = build_case((8, 7), grid=Grid((8, 7), (1.0, 1.0), num_ghost_layers))
    subdomain = build_subdomain(case.grid)
    solver = PoissonSolver(case, subdomain)
    state = build_swirl(subdomain)

    before = float(np.abs(solver.compute_divergence(state)).max())
    pressure = solver.solve(state, TIME_STEP)
    solver.correct_velocity(state, pressure, TIME_STEP)

    assert pressure.shape == subdomain.shape
    assert float(np.abs(solver.compute_divergence(state)).max()) < before


@pytest.mark.parametrize("shape", [(2,), (2, 2), (2, 2, 2)])
def test_the_fewest_points_an_axis_allows_still_projects(
    build_poisson_solver: Callable[..., tuple[PoissonSolver, Subdomain]],
    shape: tuple[int, ...],
) -> None:
    """
    Two points is the fewest Grid accepts. Each axis operator is then the two-row matrix [[-1, 1], [1, -1]], which is still symmetric and still singular, so nothing about the solve is a special case.
    """

    solver, subdomain = build_poisson_solver(shape)
    state = build_swirl(subdomain)

    before = float(np.abs(solver.compute_divergence(state)).max())
    solver.project(state, TIME_STEP)

    assert float(np.abs(solver.compute_divergence(state)).max()) < before


def test_the_residual_divergence_falls_at_second_order() -> None:
    """
    The compact Laplacian is not the exact divergence of a gradient that spans 2h, so the projection leaves a residual rather than nothing. That residual is a truncation error, so halving the spacing has to quarter it. A boundary treatment that states the wrong condition shows up here as an error that grows instead.
    """

    coarse_before, coarse_after = project_a_shear_flow(17)
    fine_before, fine_after = project_a_shear_flow(33)

    assert coarse_after < coarse_before / 10.0
    assert fine_after < fine_before / 100.0
    assert coarse_after / fine_after == pytest.approx(4.0, abs=0.5)


def project_a_shear_flow(points: int) -> tuple[float, float]:
    """
    Run one projection over a fixed physical square at `points` points per axis, and return the largest divergence before it and afterwards over the region whose stencil reads corrected values only. The field is fixed in meters, so a finer grid resolves the same flow and the pair of numbers is comparable across calls.
    """

    case = Case(
        fluid=Fluid(1.0, 0.1),
        grid=Grid((points, points), (1.0, 1.0), 1),
        time=TimeControl(1, 0.25),
    )
    subdomain = Subdomain(case.grid, ProcessGrid((1, 1)), 0)
    solver = PoissonSolver(case, subdomain)
    coordinate = (np.arange(points + 2) - 1) * case.grid.spacing[0]
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    state = allocate_state(subdomain, 2)
    state.velocity[0][...] = np.sin(2 * np.pi * x) * np.cos(np.pi * y)
    state.velocity[1][...] = np.cos(np.pi * x) * np.sin(2 * np.pi * y)

    before = float(np.abs(solver.compute_divergence(state)[1:-1, 1:-1]).max())
    solver.project(state, TIME_STEP)
    return before, float(np.abs(solver.compute_divergence(state)[1:-1, 1:-1]).max())


def test_the_projection_lowers_the_divergence_with_an_outflow_face(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> None:
    case = build_case((12, 11, 10), boundaries=Boundaries(right=FaceCondition(BoundaryCondition.OUTFLOW, pressure=0.0)))
    subdomain = build_subdomain(case.grid)
    solver = PoissonSolver(case, subdomain)
    state = build_swirl(subdomain)
    state.velocity[2][...] = 0.0

    before = float(np.abs(solver.compute_divergence(state)[1:-1, 1:-1, 1:-1]).max())
    solver.project(state, TIME_STEP)
    after = float(np.abs(solver.compute_divergence(state)[1:-1, 1:-1, 1:-1]).max())

    assert after < before / 10.0


def test_a_shifted_periodic_flow_gives_a_shifted_pressure(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> None:
    periodic = FaceCondition(BoundaryCondition.PERIODIC)
    boundaries = Boundaries(front=periodic, back=periodic)
    case = build_case((6, 5, 12), boundaries=boundaries)
    subdomain = build_subdomain(case.grid)
    exchange = GhostExchange(subdomain, boundaries, None)
    pressures = []
    for shift in (0, 3):
        state = allocate_state(subdomain, 3)
        layer = np.arange(-1, 13) - shift
        state.velocity[2][...] = (np.sin(2 * np.pi * layer / 12) + 0.3 * np.cos(4 * np.pi * layer / 12))[None, None, :]
        exchange.start(state)
        exchange.complete(state)
        pressures.append(PoissonSolver(case, subdomain).solve(state, TIME_STEP))

    assert np.abs(pressures[0]).max() > 0.1
    assert np.allclose(pressures[1], np.roll(pressures[0], 3, axis=2), atol=1e-8)


def build_channel_case(integration_order: int, include_pressure: bool) -> Case:
    periodic = FaceCondition(BoundaryCondition.PERIODIC)
    return Case(
        fluid=Fluid(1.0, 0.05),
        grid=Grid((16, 8, 8), (2.0, 1.0, 1.0), 1),
        time=TimeControl(num_steps=4, cfl=0.3, integration_order=integration_order),
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
        solver=SolverOptions(include_pressure=include_pressure),
    )


@pytest.mark.parametrize("integration_order", [1, 2, 3])
def test_a_run_with_pressure_ends_with_less_divergence(integration_order: int) -> None:
    divergence = []
    for include_pressure in (False, True):
        case = build_channel_case(integration_order, include_pressure)
        session = SimulationSession(case)
        session.run_until_complete(write_initial=False)
        measured = PoissonSolver(case, session.subdomain).compute_divergence(session.state)
        divergence.append(float(np.abs(measured[1:-1, 1:-1, 1:-1]).max()))

    assert np.isfinite(divergence).all()
    assert divergence[1] < divergence[0] / 5.0


@pytest.fixture(scope="module")
def serial_pressure_output(
    tmp_path_factory: pytest.TempPathFactory,
    run_under_mpiexec: Callable[..., None],
) -> Path:
    directory = tmp_path_factory.mktemp("serial_pressure")
    run_under_mpiexec("_run_pressure_case.py", 1, str(directory))
    return directory / "Checkpoint_Final.csv"


@pytest.mark.mpi
@pytest.mark.slow
@pytest.mark.parametrize("num_procs", [2, 4])
def test_a_pressure_run_matches_across_rank_counts(
    tmp_path: Path,
    run_under_mpiexec: Callable[..., None],
    serial_pressure_output: Path,
    num_procs: int,
) -> None:
    run_under_mpiexec("_run_pressure_case.py", num_procs, str(tmp_path))
    serial = read_checkpoint(str(serial_pressure_output))
    parallel = read_checkpoint(str(tmp_path / "Checkpoint_Final.csv"))

    assert np.allclose(parallel.velocity, serial.velocity, rtol=0.0, atol=1e-10)
    assert np.allclose(parallel.pressure, serial.pressure, rtol=0.0, atol=1e-10)
    assert np.abs(serial.pressure).max() > 0.1
