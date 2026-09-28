"""
For incompressible flow, fluid density is constant everywhere, so whatever flows into our box must flow out.
    - In other words, the net divergence is exactly 0 (grad u = 0).

This module implements Chorin's Projection Method, splitting each time step into 3 sub-steps:
    1. Predict a velocity u* using only convection/diffusion.
    2. Solve for a pressure field so that correcting u* using pressure results in gradient being 0.
    3. Fix the velocity.

The corrected velocity is: u = u* - (dt/ρ)∇p
Taking the divergence of both sides: ∇·u = ∇·u* - (dt/ρ)∇²p = 0
Rearranging: ∇²p = (ρ/dt) · ∇·u*


"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING

import numpy as np

from ..boundary_application import select_ghost_and_mirror
from ..config.boundaries import collect_faces
from ..config.boundary_conditions import BoundaryCondition
from ..config.case import Case
from ..fields import FlowState
from ..mpi.ghost_exchange import GhostExchange
from ..mpi.subdomain import Subdomain

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm


class PoissonSolver:

    def __init__(
        self,
        case: Case,
        subdomain: Subdomain,
        comm: Intracomm | None = None,
    ) -> None:
        """
        Build the discrete Laplacian for the block `subdomain` owns.

        The Laplacian is applied as a stencil and never assembled. Conjugate gradient runs on every rank at once: each iteration exchanges the ghost layers of the search direction with the neighboring ranks and sums its dot products across all of them, so every rank count reaches the same answer up to rounding. The time loop runs it through `SpatialOperator.project` when `include_pressure` is set.
        """

        # Destructure simulation parameters
        self._subdomain = subdomain
        self._comm = comm
        self._spacing = case.grid.spacing
        self._inverse_square = tuple(1.0 / (step * step) for step in case.grid.spacing)
        self._interior = subdomain.interior
        self._lower = tuple(subdomain.shift_interior(axis, -1) for axis in range(case.dimension))
        self._upper = tuple(subdomain.shift_interior(axis, +1) for axis in range(case.dimension))
        self.rho = case.fluid.rho
        self.dimension = case.dimension
        self.interior_shape = subdomain.shape
        self._num_cells = math.prod(case.grid.shape)

        self._exchange = GhostExchange(subdomain, case.boundaries, comm, num_arrays=1)
        self._faces: list[tuple[tuple[slice, ...], tuple[slice, ...], float | None]] = []
        pad = case.grid.num_ghost_layers
        for face in collect_faces(case.dimension):
            condition = case.boundaries.find_face(face)
            if not subdomain.is_on_face(face) or condition.kind == BoundaryCondition.PERIODIC:
                continue
            ghost, mirror = select_ghost_and_mirror(face, pad, case.dimension)
            fixed = condition.pressure if condition.kind == BoundaryCondition.OUTFLOW else None
            self._faces.append((ghost, mirror, fixed))
        self._has_fixed_pressure = any(
            case.boundaries.find_face(face).kind == BoundaryCondition.OUTFLOW for face in collect_faces(case.dimension)
        )

        self._padded = np.zeros(subdomain.padded_shape, dtype=float)
        self._scratch = np.empty(subdomain.shape, dtype=float)
        self._previous: np.ndarray | None = None

    # The discrete Laplacian operator ∇²p = ∂²p/∂x² + ∂²p/∂y² + ∂²p/∂z²
    # Approximated via central differences: ∂²p/∂x² ≈ (p[i+1] - 2p[i] + p[i-1]) / dx²
    def _apply_laplacian(self, padded: np.ndarray, out: np.ndarray, *, homogeneous: bool) -> None:
        self._fill_ghost_layers(padded, homogeneous=homogeneous)
        middle = padded[self._interior]
        scratch = self._scratch
        out.fill(0.0)
        for axis in range(self.dimension):
            np.add(padded[self._upper[axis]], padded[self._lower[axis]], out=scratch)
            scratch -= middle
            scratch -= middle
            scratch *= self._inverse_square[axis]
            out += scratch

    def _fill_ghost_layers(self, padded: np.ndarray, *, homogeneous: bool) -> None:
        self._exchange.start_arrays((padded,))
        self._exchange.complete_arrays((padded,))
        for ghost, mirror, fixed in self._faces:
            if fixed is None:
                padded[ghost] = padded[mirror]
            else:
                padded[ghost] = (0.0 if homogeneous else 2.0 * fixed) - padded[mirror]

    def compute_divergence(self, state: FlowState) -> np.ndarray:
        """
        Compute the divergence of the velocity field using second-order central differencing.

        Uses the stencil (u[i+1] - u[i-1]) / (2 * dx) for each direction. Returns ∇·u*, which is a term in the RHS of the pressure Poisson equation.

        Args:
            state: The velocity field to measure, ghost layers included. Read, never modified.

        Returns:
            Array of divergence values at interior grid points, shaped like the block this rank owns. The stencil reaches one point beyond each end, so the value at an outermost real point reads a ghost layer and is only as good as whatever last filled it.
        """

        divergence = np.zeros(self.interior_shape, dtype=float)
        for axis in range(self.dimension):
            field = state.velocity[axis]
            divergence += (field[self._upper[axis]] - field[self._lower[axis]]) / (2 * self._spacing[axis])
        return divergence

    def solve(self, state: FlowState, dt: float) -> np.ndarray:
        """
        Solve ∇²p = (ρ/dt) · ∇·u* for the pressure field.

        Without an outflow face, the operator carries a zero normal gradient or a periodic wrap on every face, so it is singular: it fixes the pressure only up to a constant. This then removes the mean of the right-hand side, which is the condition such a problem must satisfy to have a solution at all, and returns the one answer whose own mean is zero. An outflow face fixes the pressure there, so the answer is unique and nothing is removed.

        Args:
            state: The predicted velocity field, ghost layers included. Read, never modified.
            dt: Current timestep size.

        Returns:
            Pressure field at interior grid points (no ghost layers), shaped like the block this rank owns, with a mean of zero when no face is an outflow face.
        """
        # Step 1: Compute RHS = (ρ/dt) · ∇·u*
        div = self.compute_divergence(state)
        rhs = (self.rho / dt) * div

        # Step 2: Move the prescribed outflow pressure to the right-hand side, so the solve runs on -∇² with a zero ghost value there
        self._padded.fill(0.0)
        target = np.empty(self.interior_shape, dtype=float)
        self._apply_laplacian(self._padded, target, homogeneous=False)
        target -= rhs

        # Step 3: A pure Neumann problem is solvable only where the right-hand side sums to zero, so remove its mean.
        if not self._has_fixed_pressure:
            target -= self._compute_mean(target)

        # Step 4: Solve with conjugate gradient
        start = np.zeros(self.interior_shape, dtype=float) if self._previous is None else self._previous.copy()
        pressure = self._run_conjugate_gradient(target, start)
        self._previous = pressure.copy()

        # Step 5: Return the zero-mean answer when no face fixes the pressure
        if self._has_fixed_pressure:
            return pressure
        return pressure - self._compute_mean(pressure)

    def _run_conjugate_gradient(self, target: np.ndarray, guess: np.ndarray) -> np.ndarray:
        threshold = 1e-5 * math.sqrt(self._compute_dot(target, target))
        if threshold == 0.0:
            return np.zeros_like(target)
        product = np.empty_like(target)
        self._apply_operator(guess, product)
        residual = target - product
        direction = residual.copy()
        residual_norm = self._compute_dot(residual, residual)
        for _ in range(10 * self._num_cells):
            if math.sqrt(residual_norm) < threshold:
                return guess
            self._apply_operator(direction, product)
            step = residual_norm / self._compute_dot(direction, product)
            guess += step * direction
            residual -= step * product
            next_norm = self._compute_dot(residual, residual)
            direction *= next_norm / residual_norm
            direction += residual
            residual_norm = next_norm
        raise RuntimeError(f"Conjugate gradient did not converge in {10 * self._num_cells} iterations.")

    def _apply_operator(self, vector: np.ndarray, out: np.ndarray) -> None:
        self._padded[self._interior] = vector
        self._apply_laplacian(self._padded, out, homogeneous=True)
        np.negative(out, out=out)

    def _compute_dot(self, first: np.ndarray, second: np.ndarray) -> float:
        local = float(np.vdot(first, second))
        if self._comm is None:
            return local
        return float(self._comm.allreduce(local))

    def _compute_mean(self, array: np.ndarray) -> float:
        local = float(array.sum())
        if self._comm is not None:
            local = float(self._comm.allreduce(local))
        return local / self._num_cells

    def correct_velocity(self, state: FlowState, pressure: np.ndarray, dt: float) -> None:
        """
        Apply pressure correction: u = u* - (dt/ρ) · ∇p.

        Modifies the velocity of `state` in place at every interior point, the outermost real point included. The pressure gradient is a central difference, ∂p/∂x ≈ (p[i+1] - p[i-1]) / (2·dx), taken over a pressure array padded with the ghost layers the boundary conditions give: the mirror of the real value on a wall or inflow face, 2 * prescribed - mirror on an outflow face so that the prescribed pressure sits on the face, and the neighbor's value across a rank or periodic face. That padding states the same conditions the Laplacian carries, so the end points need no separate rule.

        The ghost layers keep the predicted velocity. The divergence at the outermost real point reads one of them, so the caller refreshes the ghost layers through the ghost exchange and the boundary conditions before that point means anything.

        The compact Laplacian is not the exact divergence of this gradient, because both differences span 2h while the operator spans h. The divergence left behind therefore falls as h^2 at a point whose stencil reads corrected values only, rather than to zero.

        Args:
            state: The predicted velocity field, ghost layers included. Modified in place.
            pressure: Pressure field at interior points (from solve()), no ghost layers.
            dt: Current timestep size.
        """
        coeff = dt / self.rho
        padded = self._padded
        padded[self._interior] = pressure
        self._fill_ghost_layers(padded, homogeneous=False)

        # ∂p/∂x_a via central differences over the padded pressure
        for axis in range(self.dimension):
            gradient = (padded[self._upper[axis]] - padded[self._lower[axis]]) / (2 * self._spacing[axis])
            state.velocity[axis][self._interior] -= coeff * gradient

    def project(self, state: FlowState, dt: float) -> None:
        """
        Make `state` as divergence free as this discretization allows, and leave the pressure that did it in `state.pressure`.

        This is the whole corrector half of Chorin's method in one call: solve for the pressure, correct the velocity with its gradient, and record the pressure. The caller runs it after a time-integration stage has produced the predicted velocity.
        """

        pressure = self.solve(state, dt)
        self.correct_velocity(state, pressure, dt)
        state.pressure[self._interior] = pressure
