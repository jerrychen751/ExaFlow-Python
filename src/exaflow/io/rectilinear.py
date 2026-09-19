from __future__ import annotations

import warnings

import numpy as np
import pyvista as pv

from ..config.grid import Grid


def build_rectilinear_grid(
    grid: Grid,
    components: list[np.ndarray],
    pressure: np.ndarray,
) -> pv.RectilinearGrid:
    axes = [np.linspace(0.0, float(span), int(count)) for span, count in zip(grid.extent, grid.shape)]
    while len(axes) < 3:
        axes.append(np.array([0.0], dtype=float))
    padded = pressure.shape + (1,) * (3 - pressure.ndim)
    pressure = np.ascontiguousarray(pressure.reshape(padded))
    components = [np.ascontiguousarray(part.reshape(padded)) for part in components]
    while len(components) < 3:
        components.append(np.zeros(padded))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated",
            category=DeprecationWarning,
        )
        dataset = pv.RectilinearGrid(*axes)
        dataset.point_data["pressure"] = pressure.ravel(order="F")
        dataset.point_data["velocity"] = np.column_stack([part.ravel(order="F") for part in components])
    return dataset
