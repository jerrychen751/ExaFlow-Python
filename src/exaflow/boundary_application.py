from __future__ import annotations

from .config.boundary_conditions import BoundaryCondition
from .config.boundaries import Face, collect_faces
from .config.case import Case
from .fields import FlowState
from .mpi.subdomain import Subdomain


def initialize_boundaries(state: FlowState, case: Case, subdomain: Subdomain) -> None:
    """
    Write the prescribed values into the ghost layers of every global domain face this rank owns. Call once, before the first step.

    A face this rank does not own is skipped: its ghost layer belongs to the ghost exchange, not to a boundary condition. A periodic face is skipped for the same reason, because the exchange wraps it.
    """

    pad = case.grid.num_ghost_layers
    for face in collect_faces(case.dimension):
        if not subdomain.is_on_face(face):
            continue
        condition = case.boundaries.find_face(face)
        ghost = _select(face, _build_ghost_span(face, pad), case.dimension)

        match condition.kind:
            case BoundaryCondition.NO_SLIP:
                for axis in range(case.dimension):
                    state.velocity[axis][ghost] = 0.0
            case BoundaryCondition.SLIP:
                state.velocity[face.axis][ghost] = 0.0
            case BoundaryCondition.INFLOW:
                for axis in range(case.dimension):
                    state.velocity[axis][ghost] = condition.velocity[axis]
            case BoundaryCondition.OUTFLOW:
                state.pressure[ghost] = condition.pressure
            case BoundaryCondition.PERIODIC:
                pass
            case _:
                raise ValueError(f"Unsupported boundary condition on {face.name}: {condition.kind}.")


def update_boundaries(state: FlowState, case: Case, subdomain: Subdomain) -> None:
    """
    Refresh the zero-gradient half of each boundary condition in the ghost layers of every global domain face this rank owns. Call once per stage, after the ghost exchange completes.

    A wall or inflow face fixes velocity and lets pressure float, so pressure is copied outward from the first real layer. A slip wall fixes only the velocity component normal to it, so the tangential components are copied outward as well. An outflow face fixes pressure and lets velocity float, so the velocity components are copied instead.

    The source span is kept one element wide so that assigning it into a ghost region of depth `pad` broadcasts over the whole depth.
    """

    pad = case.grid.num_ghost_layers
    for face in collect_faces(case.dimension):
        if not subdomain.is_on_face(face):
            continue
        condition = case.boundaries.find_face(face)
        ghost = _select(face, _build_ghost_span(face, pad), case.dimension)
        edge_span = slice(pad, pad + 1) if face.is_low else slice(-pad - 1, -pad)
        edge = _select(face, edge_span, case.dimension)

        match condition.kind:
            case BoundaryCondition.NO_SLIP | BoundaryCondition.INFLOW:
                state.pressure[ghost] = state.pressure[edge]
            case BoundaryCondition.SLIP:
                state.pressure[ghost] = state.pressure[edge]
                for axis in range(case.dimension):
                    if axis != face.axis:
                        state.velocity[axis][ghost] = state.velocity[axis][edge]
            case BoundaryCondition.OUTFLOW:
                for axis in range(case.dimension):
                    state.velocity[axis][ghost] = state.velocity[axis][edge]
            case BoundaryCondition.PERIODIC:
                pass
            case _:
                raise ValueError(f"Unsupported boundary condition on {face.name}: {condition.kind}.")


def _build_ghost_span(face: Face, pad: int) -> slice:
    return slice(0, pad) if face.is_low else slice(-pad, None)


def _select(face: Face, span: slice, dimension: int) -> tuple[slice, ...]:
    return tuple(span if axis == face.axis else slice(None) for axis in range(dimension))
