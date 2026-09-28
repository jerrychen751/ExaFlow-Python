from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..boundary_application import update_boundaries
from ..config.case import Case
from ..fields import FlowState
from ..mpi.ghost_exchange import GhostExchange
from ..mpi.subdomain import Subdomain
from .convection import Convection
from .diffusion import Diffusion
from .pressure_poisson import PoissonSolver

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm


class Operator(Protocol):
    """
    One term of the right-hand side. `accumulate` adds this term's contribution to `rate` for every real cell and must not clear what another term already wrote.
    """

    def accumulate(self, state: FlowState, rate: FlowState) -> None: ...


def build_operators(case: Case, subdomain: Subdomain) -> tuple[Operator, ...]:
    """
    The terms this case switches on, in the order they are accumulated. Each one is bound to this rank's block, so the dimension and the index rules are settled once here rather than at every call.
    """

    operators: list[Operator] = []
    if case.solver.include_convection:
        operators.append(Convection(case.grid, subdomain))
    if case.solver.include_diffusion:
        operators.append(Diffusion(case.fluid, case.grid, subdomain))
    return tuple(operators)


class SpatialOperator:
    """
    The whole right-hand side for one Runge-Kutta stage: refresh the ghost layers, refresh the boundary conditions, then accumulate every enabled term.

    Every stencil here reads the ghost layer of the outermost real cell, so the exchange is completed before any term runs. That costs the overlap of communication with interior work that a peeled bulk-then-shell traversal would allow; the peeled form is worth adding once a profile shows the exchange is the limit.
    """

    def __init__(self, case: Case, subdomain: Subdomain, comm: Intracomm | None) -> None:
        self._case = case
        self._subdomain = subdomain
        self._exchange = GhostExchange(subdomain, case.boundaries, comm)
        self._operators = build_operators(case, subdomain)
        self._poisson = PoissonSolver(case, subdomain, comm) if case.solver.include_pressure else None

    def evaluate(self, state: FlowState, rate: FlowState) -> None:
        """
        Overwrite `rate` with the right-hand side at `state`. The rate excludes the time step, so an integrator scales it by dt. `state` is mutated in its ghost layers only.
        """

        rate.velocity.fill(0.0)
        rate.pressure.fill(0.0)
        self._refresh_ghost_layers(state)
        for operator in self._operators:
            operator.accumulate(state, rate)

    def project(self, state: FlowState, dt: float) -> None:
        if self._poisson is None:
            return
        self._refresh_ghost_layers(state)
        self._poisson.project(state, dt)

    def _refresh_ghost_layers(self, state: FlowState) -> None:
        self._exchange.start(state)
        self._exchange.complete(state)
        update_boundaries(state, self._case, self._subdomain)
