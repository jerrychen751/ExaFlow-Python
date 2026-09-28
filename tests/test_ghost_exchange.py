from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from exaflow.config import Boundaries, BoundaryCondition, FaceCondition, Grid
from exaflow.fields import allocate_state
from exaflow.mpi.ghost_exchange import GhostExchange
from exaflow.mpi.process_grid import ProcessGrid
from exaflow.mpi.subdomain import Subdomain

PERIODIC = FaceCondition(BoundaryCondition.PERIODIC)


def test_a_periodic_axis_on_one_rank_copies_its_own_opposite_face(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    """
    A periodic axis carrying a single rank makes that rank its own neighbor. The exchange serves it with a local copy instead of a message, so a serial periodic run needs no communicator.
    """

    subdomain = build_subdomain(Grid((8,), (1.0,), 1))
    state = allocate_state(subdomain, 1)
    state.velocity[0][subdomain.interior] = np.arange(8.0)
    exchange = GhostExchange(subdomain, Boundaries(left=PERIODIC, right=PERIODIC), None)

    exchange.start(state)
    exchange.complete(state)

    assert state.velocity[0][0] == 7.0
    assert state.velocity[0][-1] == 0.0


def test_a_single_array_exchange_wraps_a_periodic_axis(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    subdomain = build_subdomain(Grid((8,), (1.0,), 1))
    field = np.zeros(subdomain.padded_shape)
    field[subdomain.interior] = np.arange(8.0)
    exchange = GhostExchange(subdomain, Boundaries(left=PERIODIC, right=PERIODIC), None, num_arrays=1)

    exchange.start_arrays((field,))
    exchange.complete_arrays((field,))

    assert field[0] == 7.0
    assert field[-1] == 0.0


def test_the_exchange_carries_pressure_as_well_as_velocity(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    subdomain = build_subdomain(Grid((8,), (1.0,), 1))
    state = allocate_state(subdomain, 1)
    state.pressure[subdomain.interior] = np.arange(10.0, 18.0)
    exchange = GhostExchange(subdomain, Boundaries(left=PERIODIC, right=PERIODIC), None)

    exchange.start(state)
    exchange.complete(state)

    assert state.pressure[0] == 17.0
    assert state.pressure[-1] == 10.0


def test_a_face_with_no_neighbor_is_left_to_the_boundary_conditions(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    subdomain = build_subdomain(Grid((8,), (1.0,), 1))
    state = allocate_state(subdomain, 1)
    state.velocity[...] = 4.0
    exchange = GhostExchange(subdomain, Boundaries(), None)

    exchange.start(state)
    exchange.complete(state)

    assert np.all(state.velocity == 4.0)


def test_only_the_interior_transverse_extent_crosses_a_periodic_face(
    build_subdomain: Callable[..., Subdomain],
) -> None:
    """
    The corner region of a ghost plane is left alone, because no stencil in this solver reads a diagonal ghost cell. A stencil that did would need a second pass, one axis at a time.
    """

    subdomain = build_subdomain(Grid((6, 5), (1.0, 1.0), 1))
    state = allocate_state(subdomain, 2)
    state.velocity[...] = -1.0
    state.velocity[0][subdomain.interior] = 2.0
    exchange = GhostExchange(subdomain, Boundaries(left=PERIODIC, right=PERIODIC), None)

    exchange.start(state)
    exchange.complete(state)

    assert np.all(state.velocity[0][0, subdomain.interior[1]] == 2.0)
    assert state.velocity[0][0, 0] == -1.0
    assert state.velocity[0][0, -1] == -1.0


def test_a_neighbor_on_another_rank_needs_a_communicator() -> None:
    """
    The constructor raises here rather than at the first stage, so a driver built without the communicator fails before it marches.
    """

    grid = Grid((8,), (1.0,), 1)
    subdomain = Subdomain(grid, ProcessGrid((2,)), 0)
    with pytest.raises(ValueError, match="has neighbor rank 1 but no communicator"):
        GhostExchange(subdomain, Boundaries(), None)


@pytest.mark.mpi
@pytest.mark.slow
@pytest.mark.parametrize("num_procs", [1, 2, 4])
def test_a_periodic_axis_wraps_onto_the_right_plane(
    run_under_mpiexec: Callable[..., None],
    num_procs: int,
) -> None:
    """
    Two ranks on a periodic axis are each other's neighbor on both faces, so each posts two sends and two receives to the same peer. Untagged, MPI matched them in order and filled the low ghost plane with the high one, which no rank-count independence test caught.
    """

    run_under_mpiexec("_run_periodic_exchange.py", num_procs)
