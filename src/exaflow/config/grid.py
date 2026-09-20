from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class Grid:
    """
    The global structured grid.

    A grid point sits on both ends of each span, so spacing is extent / (points - 1). Every axis therefore needs at least two points.
    """

    shape: tuple[int, ...]  # (nx, ny, nz) grid points per axis; its length fixes the dimension of the whole case
    extent: tuple[float, ...]  # physical span per axis in meters, in the same axis order as shape
    num_ghost_layers: int = 1  # pad depth added at each end of every axis of a rank-local array

    def __post_init__(self) -> None:
        if len(self.shape) not in (1, 2, 3):
            raise ValueError(f"shape must have 1, 2 or 3 axes, got {self.shape!r}.")
        if len(self.extent) != len(self.shape):
            raise ValueError(f"extent must match shape in length, got shape={self.shape!r}, extent={self.extent!r}.")
        if any(count < 2 for count in self.shape):
            raise ValueError(f"every axis needs at least 2 grid points, got {self.shape!r}.")
        for span in self.extent:
            if not math.isfinite(span) or span <= 0.0:
                raise ValueError(f"every extent must be finite and > 0, got {self.extent!r}.")
        if self.num_ghost_layers < 1:
            raise ValueError(f"num_ghost_layers must be >= 1, got {self.num_ghost_layers}.")

    @property
    def dimension(self) -> int:
        return len(self.shape)

    @property
    def spacing(self) -> tuple[float, ...]:
        """
        Grid spacing per axis in meters, as (dx, dy, dz) truncated to the dimension of this grid.
        """

        return tuple(span / float(count - 1) for span, count in zip(self.extent, self.shape))
