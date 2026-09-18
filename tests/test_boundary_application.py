from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from exaflow.boundary_application import initialize_boundaries, update_boundaries
from exaflow.config import Boundaries, BoundaryCondition, Case, FaceCondition, Grid
from exaflow.fields import FlowState, allocate_state
from exaflow.mpi.process_grid import ProcessGrid
from exaflow.mpi.subdomain import Subdomain

PERIODIC = FaceCondition(BoundaryCondition.PERIODIC)
FILLER = 9.0


def isolate(condition: FaceCondition, face_name: str = "left") -> Boundaries:
    """
    A 2D boundary record that carries `condition` on the named face and wraps the y axis, so no other face writes into the plane under test. Every face is applied in turn and a later face overwrites the transverse corners of an earlier one.
    """

    return Boundaries(top=PERIODIC, bottom=PERIODIC, **{face_name: condition})


def build_filled_state(case: Case, subdomain: Subdomain) -> FlowState:
    """
    A state whose every cell holds FILLER, so any cell a boundary condition writes is visible against the cells it must leave alone.
    """

    state = allocate_state(subdomain, case.dimension)
    state.velocity[...] = FILLER
    state.pressure[...] = FILLER
    return state


@pytest.fixture
def apply_initial(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> Callable[..., tuple[FlowState, Subdomain]]:
    def run(boundaries: Boundaries, num_ghost_layers: int = 1) -> tuple[FlowState, Subdomain]:
        case = build_case(
            (6, 5),
            grid=Grid((6, 5), (1.0, 1.0), num_ghost_layers),
            boundaries=boundaries,
        )
        subdomain = build_subdomain(case.grid)
        state = build_filled_state(case, subdomain)
        initialize_boundaries(state, case, subdomain)
        return state, subdomain

    return run


def test_a_no_slip_face_stops_every_velocity_component(
    apply_initial: Callable[..., tuple[FlowState, Subdomain]],
) -> None:
    state, subdomain = apply_initial(isolate(FaceCondition(BoundaryCondition.NO_SLIP)))
    transverse = subdomain.interior[1]
    assert np.all(state.velocity[0][0, transverse] == 0.0)
    assert np.all(state.velocity[1][0, transverse] == 0.0)


def test_a_slip_face_stops_only_the_component_normal_to_it(
    apply_initial: Callable[..., tuple[FlowState, Subdomain]],
) -> None:
    """
    A slip wall lets the flow run along it, so the tangential components keep whatever the ghost layer held.
    """

    state, subdomain = apply_initial(isolate(FaceCondition(BoundaryCondition.SLIP)))
    transverse = subdomain.interior[1]
    assert np.all(state.velocity[0][0, transverse] == 0.0)
    assert np.all(state.velocity[1][0, transverse] == FILLER)


def test_an_inflow_face_writes_each_prescribed_component(
    apply_initial: Callable[..., tuple[FlowState, Subdomain]],
) -> None:
    state, subdomain = apply_initial(isolate(FaceCondition(BoundaryCondition.INFLOW, (1.5, -0.5))))
    transverse = subdomain.interior[1]
    assert np.all(state.velocity[0][0, transverse] == 1.5)
    assert np.all(state.velocity[1][0, transverse] == -0.5)


def test_an_outflow_face_writes_the_pressure_and_leaves_the_velocity(
    apply_initial: Callable[..., tuple[FlowState, Subdomain]],
) -> None:
    state, subdomain = apply_initial(
        isolate(FaceCondition(BoundaryCondition.OUTFLOW, pressure=2.25), "right")
    )
    transverse = subdomain.interior[1]
    assert np.all(state.pressure[-1, transverse] == 2.25)
    assert np.all(state.velocity[0][-1, transverse] == FILLER)


def test_a_periodic_face_is_left_to_the_ghost_exchange(
    apply_initial: Callable[..., tuple[FlowState, Subdomain]],
) -> None:
    state, _ = apply_initial(Boundaries(left=PERIODIC, right=PERIODIC, top=PERIODIC, bottom=PERIODIC))
    assert np.all(state.velocity == FILLER)
    assert np.all(state.pressure == FILLER)


def test_the_prescribed_value_fills_every_ghost_layer(
    apply_initial: Callable[..., tuple[FlowState, Subdomain]],
) -> None:
    state, subdomain = apply_initial(isolate(FaceCondition(BoundaryCondition.INFLOW, (1.5, -0.5))), 3)
    transverse = subdomain.interior[1]
    assert np.all(state.velocity[0][:3, transverse] == 1.5)
    assert np.all(state.velocity[0][3, transverse] == FILLER)


def test_a_face_this_rank_does_not_own_keeps_the_data_of_its_neighbor(
    build_case: Callable[..., Case],
) -> None:
    """
    The cells beyond an internal partition face belong to another rank, and the ghost exchange fills them. A boundary condition applied there would overwrite real data with a wall.
    """

    case = build_case((6, 5), boundaries=isolate(FaceCondition(BoundaryCondition.NO_SLIP)))
    subdomain = Subdomain(case.grid, ProcessGrid((2, 1)), 1)
    state = build_filled_state(case, subdomain)

    initialize_boundaries(state, case, subdomain)

    assert np.all(state.velocity[0][0, subdomain.interior[1]] == FILLER)


@pytest.mark.parametrize(
    "condition",
    [
        FaceCondition(BoundaryCondition.NO_SLIP),
        FaceCondition(BoundaryCondition.SLIP),
        FaceCondition(BoundaryCondition.INFLOW, (1.0, 1.0)),
    ],
)
def test_a_wall_or_inflow_face_copies_the_pressure_outward(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
    condition: FaceCondition,
) -> None:
    """
    A face that fixes velocity lets pressure float, so the ghost layer takes the value of the first real layer. That gives a zero-gradient pressure across the face.
    """

    case = build_case((6, 5), boundaries=isolate(condition))
    subdomain = build_subdomain(case.grid)
    state = allocate_state(subdomain, case.dimension)
    state.pressure[subdomain.interior] = np.arange(30.0).reshape(6, 5)  # (30,) -> (6, 5)

    update_boundaries(state, case, subdomain)

    assert np.all(state.pressure[0, subdomain.interior[1]] == state.pressure[1, subdomain.interior[1]])
    assert state.pressure[0, 1] == 0.0


def test_an_outflow_face_copies_the_velocity_outward(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> None:
    case = build_case(
        (6, 5),
        boundaries=isolate(FaceCondition(BoundaryCondition.OUTFLOW, pressure=1.0), "right"),
    )
    subdomain = build_subdomain(case.grid)
    state = allocate_state(subdomain, case.dimension)
    state.velocity[0][subdomain.interior] = np.arange(30.0).reshape(6, 5)  # (30,) -> (6, 5)

    update_boundaries(state, case, subdomain)

    transverse = subdomain.interior[1]
    assert np.all(state.velocity[0][-1, transverse] == state.velocity[0][-2, transverse])
    assert state.velocity[0][-1, 1] == 25.0


def test_a_slip_face_copies_the_tangential_velocity_outward_and_keeps_the_normal_one_stopped(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> None:
    case = build_case((6, 5), boundaries=isolate(FaceCondition(BoundaryCondition.SLIP)))
    subdomain = build_subdomain(case.grid)
    state = allocate_state(subdomain, case.dimension)
    initialize_boundaries(state, case, subdomain)
    for axis in range(2):
        state.velocity[axis][subdomain.interior] = np.arange(1.0, 31.0).reshape(6, 5)  # (30,) -> (6, 5)

    update_boundaries(state, case, subdomain)

    transverse = subdomain.interior[1]
    assert np.all(state.velocity[1][0, transverse] == state.velocity[1][1, transverse])
    assert state.velocity[1][0, 1] == 1.0
    assert np.all(state.velocity[0][0, transverse] == 0.0)


def test_the_outward_copy_reaches_every_ghost_layer(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> None:
    """
    The source span is one layer wide so that it broadcasts over the whole pad. A source as deep as the pad would reverse the layer order.
    """

    case = build_case(
        (6, 5),
        grid=Grid((6, 5), (1.0, 1.0), 3),
        boundaries=isolate(FaceCondition(BoundaryCondition.NO_SLIP)),
    )
    subdomain = build_subdomain(case.grid)
    state = allocate_state(subdomain, case.dimension)
    state.pressure[subdomain.interior] = np.arange(30.0).reshape(6, 5)  # (30,) -> (6, 5)

    update_boundaries(state, case, subdomain)

    transverse = subdomain.interior[1]
    expected = state.pressure[3, transverse]
    assert all(np.array_equal(state.pressure[layer, transverse], expected) for layer in range(3))
