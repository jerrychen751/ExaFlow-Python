from __future__ import annotations

from .config.boundary_conditions import BoundaryCondition
from .config.boundaries import Face, collect_faces
from .config.case import Case
from .fields import FlowState
from .mpi.subdomain import Subdomain


def update_boundaries(state: FlowState, case: Case, subdomain: Subdomain) -> None:
    """
    Write the ghost layers of every global domain face this rank owns. Call once before the first step, and once per stage after the ghost exchange completes.

    Each grid point is the center of a cell, so a domain face lies half a spacing beyond the outermost real point. Every ghost layer mirrors the real layer at the same distance from the face. A value the condition prescribes is written as 2 * prescribed - mirror, which puts the prescribed value exactly on the face. A value the condition leaves free is written equal to its mirror, which gives it a zero gradient across the face.

    A wall or inflow face prescribes velocity and lets pressure float. A slip wall prescribes only the velocity component normal to it, which is zero, and lets the tangential components float. An outflow face prescribes pressure and lets velocity float.

    A face this rank does not own is skipped: its ghost layer belongs to the ghost exchange, not to a boundary condition. A periodic face is skipped for the same reason, because the exchange wraps it.
    """

    pad = case.grid.num_ghost_layers
    for face in collect_faces(case.dimension):
        if not subdomain.is_on_face(face):
            continue
        condition = case.boundaries.find_face(face)
        ghost, mirror = select_ghost_and_mirror(face, pad, case.dimension)

        match condition.kind:
            case BoundaryCondition.NO_SLIP:
                for axis in range(case.dimension):
                    state.velocity[axis][ghost] = -state.velocity[axis][mirror]
                state.pressure[ghost] = state.pressure[mirror]
            case BoundaryCondition.SLIP:
                for axis in range(case.dimension):
                    sign = -1.0 if axis == face.axis else 1.0
                    state.velocity[axis][ghost] = sign * state.velocity[axis][mirror]
                state.pressure[ghost] = state.pressure[mirror]
            case BoundaryCondition.INFLOW:
                for axis in range(case.dimension):
                    state.velocity[axis][ghost] = 2.0 * condition.velocity[axis] - state.velocity[axis][mirror]
                state.pressure[ghost] = state.pressure[mirror]
            case BoundaryCondition.OUTFLOW:
                for axis in range(case.dimension):
                    state.velocity[axis][ghost] = state.velocity[axis][mirror]
                state.pressure[ghost] = 2.0 * condition.pressure - state.pressure[mirror]
            case BoundaryCondition.PERIODIC:
                pass
            case _:
                raise ValueError(f"Unsupported boundary condition on {face.name}: {condition.kind}.")


def select_ghost_and_mirror(face: Face, pad: int, dimension: int) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    if face.is_low:
        ghost_span, mirror_span = slice(0, pad), slice(2 * pad - 1, pad - 1, -1)
    else:
        ghost_span, mirror_span = slice(-pad, None), slice(-pad - 1, -2 * pad - 1, -1)
    ghost = tuple(ghost_span if axis == face.axis else slice(None) for axis in range(dimension))
    mirror = tuple(mirror_span if axis == face.axis else slice(None) for axis in range(dimension))
    return ghost, mirror
