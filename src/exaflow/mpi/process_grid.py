from __future__ import annotations

from dataclasses import dataclass
import math


def choose_process_grid(num_procs: int, shape: tuple[int, ...], num_ghost_layers: int = 1) -> tuple[int, ...]:
    """
    Split `num_procs` ranks over the axes of a grid of this shape, so that the ranks per axis track the grid points per axis as closely as the factors of `num_procs` allow. Returns one count per axis, and the counts multiply to `num_procs`.

    Every factorization of `num_procs` is searched, so a long thin grid can put every rank on its long axis. A split that would leave any block thinner than `num_ghost_layers` is rejected, because the ghost exchange reads that many real layers from each end of a block. When no split survives, this raises rather than return a decomposition that `Subdomain` would refuse.
    """

    if num_procs < 1:
        raise ValueError(f"num_procs must be >= 1, got {num_procs}.")

    candidates: list[tuple[int, ...]] = [()]
    for _ in range(len(shape) - 1):
        candidates = [
            (*prefix, count)
            for prefix in candidates
            for count in range(1, num_procs // math.prod(prefix) + 1)
            if (num_procs // math.prod(prefix)) % count == 0
        ]
    candidates = [(*prefix, num_procs // math.prod(prefix)) for prefix in candidates]

    best: tuple[int, ...] | None = None
    best_error = float("inf")
    for counts in candidates:
        if any(points // count < num_ghost_layers for count, points in zip(counts, shape)):
            continue
        density = [count / points for count, points in zip(counts, shape)]
        error = sum(abs(a - b) for i, a in enumerate(density) for b in density[i + 1 :])
        if error < best_error:
            best_error = error
            best = counts

    if best is None:
        raise ValueError(
            f"{num_procs} ranks cannot be split over a grid of shape {shape} with {num_ghost_layers} "
            f"ghost layers: every factorization leaves some rank with a block thinner than the ghost layers."
        )
    return best


@dataclass(frozen=True, slots=True)
class ProcessGrid:
    """How the MPI ranks are arranged over the domain axes."""

    counts: tuple[int, ...]  # ranks along each axis, in axis order; multiplies to the communicator size

    def __post_init__(self) -> None:
        if not self.counts or any(count < 1 for count in self.counts):
            raise ValueError(f"every axis needs at least one rank, got {self.counts!r}.")

    @property
    def size(self) -> int:
        return math.prod(self.counts)

    @property
    def dimension(self) -> int:
        return len(self.counts)

    def compute_coords(self, rank: int) -> tuple[int, ...]:
        """
        The position of this rank in the process grid, in axis order.
        """

        if not 0 <= rank < self.size:
            raise ValueError(f"rank must lie in [0, {self.size}), got {rank}.")
        coords = []
        remaining = rank
        for count in self.counts:
            coords.append(remaining % count)
            remaining //= count
        return tuple(coords)

    def compute_rank(self, coords: tuple[int, ...]) -> int:
        """
        The rank at this position in the process grid. The caller must keep every coordinate inside the grid; wrapping is the caller's job.
        """

        rank = 0
        stride = 1
        for count, index in zip(self.counts, coords):
            if not 0 <= index < count:
                raise ValueError(f"coordinate {index} is outside an axis of {count} ranks.")
            rank += index * stride
            stride *= count
        return rank
