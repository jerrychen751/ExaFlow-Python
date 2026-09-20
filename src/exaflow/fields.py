from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config.case import Case
from .config.initial_conditions import FieldInitial, StepValue, UniformValue
from .mpi.subdomain import Subdomain


@dataclass(frozen=True, slots=True)
class TimeLevel:
    """
    Where a run had reached when a state was taken. Every file a run writes carries these three values, and a restart reads them back and marches on at `dt`. A run that stops on an end time cuts its last step to the time that is left; `dt` still reports the size the run marches at, because a restart that took the cut size would keep it for every later step.
    """

    step_index: int  # completed steps, counted from time zero
    current_time: float  # simulated time in seconds at step_index
    dt: float  # step size in seconds the run marches at


@dataclass(slots=True)
class FlowState:
    """
    The velocity and pressure fields one rank holds at one time level, including ghost layers.

    Both arrays are float64. Callers mutate these arrays in place, so a state handed to an operator as a source must not be the same object as the destination.
    """

    velocity: np.ndarray  # (dimension, *padded_shape); component a is the velocity along axis a, and each component is a contiguous slab
    pressure: np.ndarray  # (*padded_shape,)

    @property
    def dimension(self) -> int:
        return int(self.velocity.shape[0])

    @property
    def padded_shape(self) -> tuple[int, ...]:
        return tuple(int(length) for length in self.pressure.shape)

    def collect_arrays(self) -> tuple[np.ndarray, ...]:
        """
        Every field array, velocity components in axis order and then pressure. This is the order the ghost exchange and the writers walk them in.
        """

        return (*(self.velocity[axis] for axis in range(self.dimension)), self.pressure)

    def copy(self) -> FlowState:
        return FlowState(velocity=self.velocity.copy(), pressure=self.pressure.copy())

    def allocate_zeros(self) -> FlowState:
        return FlowState(velocity=np.zeros_like(self.velocity), pressure=np.zeros_like(self.pressure))

    def set_sum(self, base: FlowState, rate: FlowState, factor: float) -> None:
        """
        Set this state to base + factor * rate, in place. `rate` may be this same object; `base` must not be, because the first write would destroy it before it is read.
        """

        np.multiply(rate.velocity, factor, out=self.velocity)
        self.velocity += base.velocity
        np.multiply(rate.pressure, factor, out=self.pressure)
        self.pressure += base.pressure

    def set_blend(self, first: FlowState, second: FlowState, weight: float) -> None:
        """
        Set this state to (1 - weight) * first + weight * second, in place. `second` may be this same object, which is what the final stage of the third-order scheme needs; `first` must not be, because the first write would destroy it before it is read.
        """

        np.multiply(second.velocity, weight, out=self.velocity)
        self.velocity += (1.0 - weight) * first.velocity
        np.multiply(second.pressure, weight, out=self.pressure)
        self.pressure += (1.0 - weight) * first.pressure

    def compute_max_speed(self) -> float:
        """
        The largest velocity magnitude sqrt(sum over axes of u_a^2) anywhere in this rank's arrays, ghost layers included. This is a local value; the caller reduces it across ranks.

        The magnitude, not the largest single component: a cell with u = v = w = 1 travels at sqrt(3), so a CFL limit divided by the component would be sqrt(3) too large and the run would diverge while reporting a stable step.
        """

        if self.velocity.size == 0:
            return 0.0
        return float(np.sqrt(np.einsum("d...,d...->...", self.velocity, self.velocity).max()))


def allocate_state(subdomain: Subdomain, dimension: int) -> FlowState:
    """
    A zero-filled state shaped for this rank's block, ghost layers included.
    """

    padded = subdomain.padded_shape
    return FlowState(
        velocity=np.zeros((dimension, *padded), dtype=float),
        pressure=np.zeros(padded, dtype=float),
    )


def build_initial_state(case: Case, subdomain: Subdomain) -> FlowState:
    """
    Evaluate the case initial conditions over the grid points this rank owns. Every contribution is a function of the global grid index, so a rank fills its own block directly and no full-domain array is ever built.
    """

    state = allocate_state(subdomain, case.dimension)
    _apply_contributions(case.initial.pressure, state.pressure, case, subdomain)
    for axis in range(case.dimension):
        contributions: FieldInitial = ()
        if case.initial.velocity:
            contributions = case.initial.velocity[axis]
        _apply_contributions(contributions, state.velocity[axis], case, subdomain)
    return state


def _apply_contributions(
    contributions: FieldInitial,
    array: np.ndarray,
    case: Case,
    subdomain: Subdomain,
) -> None:
    interior = array[subdomain.interior]
    for contribution in contributions:
        if isinstance(contribution, UniformValue):
            interior += contribution.value
            continue
        if isinstance(contribution, StepValue):
            local = _intersect_step(contribution, case, subdomain)
            if local is not None:
                interior[local] += contribution.magnitude
            continue
        raise TypeError(f"Unsupported initial contribution: {contribution!r}")


def _intersect_step(step: StepValue, case: Case, subdomain: Subdomain) -> tuple[slice, ...] | None:
    """
    Where the step box lands inside this rank's interior, or None when the box misses this rank. The global bounds follow ceil(start * n) <= i <= floor(end * n), which is the rule the input XML documents.
    """

    local_slices = []
    for axis, (points, (block_start, block_stop)) in enumerate(zip(case.grid.shape, subdomain.bounds)):
        first = int(math.ceil(step.start[axis] * points))
        last = int(math.floor(step.end[axis] * points))
        low = max(first, block_start)
        high = min(last + 1, block_stop)
        if high <= low:
            return None
        local_slices.append(slice(low - block_start, high - block_start))
    return tuple(local_slices)
