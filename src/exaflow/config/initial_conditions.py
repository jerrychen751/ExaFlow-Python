from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias
import math


@dataclass(frozen=True, slots=True)
class UniformValue:
    """
    Add the same constant to every grid point.
    """

    value: float


@dataclass(frozen=True, slots=True)
class StepValue:
    """
    Add a constant inside an axis-aligned box and nothing outside it.
    """

    magnitude: float  # the constant added inside the box
    start: tuple[float, ...]  # low box bound per axis as a fraction of that axis span, inclusive, within [0, 1]
    end: tuple[float, ...]  # high box bound per axis as a fraction of that axis span, inclusive, within [0, 1] and >= start

    def __post_init__(self) -> None:
        if len(self.start) != len(self.end):
            raise ValueError(f"start and end must match in length, got {self.start!r} and {self.end!r}.")
        for low, high in zip(self.start, self.end):
            if not (math.isfinite(low) and math.isfinite(high)):
                raise ValueError(f"step bounds must be finite, got {self.start!r} and {self.end!r}.")
            if not (0.0 <= low <= 1.0 and 0.0 <= high <= 1.0):
                raise ValueError(f"step bounds must lie within [0, 1], got start={low}, end={high}.")
            if high < low:
                raise ValueError(f"step end must be >= start, got start={low}, end={high}.")


FieldInitial: TypeAlias = tuple[UniformValue | StepValue, ...]


@dataclass(frozen=True, slots=True)
class InitialConditions:
    """
    The starting field for each quantity, given as contributions that are summed in order. An empty tuple leaves that quantity at zero.
    """

    pressure: FieldInitial = ()
    velocity: tuple[FieldInitial, ...] = ()  # one entry per axis, in axis order
