from __future__ import annotations

import os

import numpy as np

from ..fields import TimeLevel


def format_field_csv(
    velocity: np.ndarray,
    pressure: np.ndarray,
    origin: tuple[int, ...],
    level: TimeLevel,
) -> str:
    """
    Render one block as CSV text. `velocity` is (dimension, *shape) and `pressure` is (shape). `origin` is the global index of the block's first cell, so the index columns carry global indices even when the block is one rank's share. `level` states where the run had reached, in seconds.

    The first line starts with `#` and names the step, the simulated time and the step size, so a file states its own place in the run. The column header follows it, then the rows: the axis indices, the velocity components in axis order, then pressure, with the last axis varying fastest. Every value is written at full float64 precision, so the text reads back as the array that produced it.
    """

    dimension = int(velocity.shape[0])
    stamp = f"# step={level.step_index} time={level.current_time!r} dt={level.dt!r}"
    header = ",".join((*("x", "y", "z")[:dimension], *("u", "v", "w")[:dimension], "p"))

    indices = np.indices(pressure.shape).reshape(dimension, -1)  # (dimension, *shape) -> (dimension, N)
    indices += np.asarray(origin, dtype=indices.dtype)[:, None]  # (dimension,) -> (dimension, 1) broadcast over N
    columns = [
        *(axis.tolist() for axis in indices),
        *(component.tolist() for component in velocity.reshape(dimension, -1)),  # (dimension, *shape) -> (dimension, N)
        pressure.reshape(-1).tolist(),  # (*shape,) -> (N,)
    ]

    template = ", ".join(["%d"] * dimension + ["%r"] * (dimension + 1))
    rows = [template % row for row in zip(*columns)]
    return f"{stamp}\n{header}\n" + "\n".join(rows) + "\n"


def write_text_atomically(path: str, contents: str) -> None:
    """
    Write the file so that a reader never sees a half-written one. The text goes to a sibling `.partial` file, is flushed to disk, and is then renamed over the target.
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    partial_path = f"{path}.partial"
    with open(partial_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial_path, path)
