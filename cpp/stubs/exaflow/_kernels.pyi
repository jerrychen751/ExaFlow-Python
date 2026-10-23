"""
Define Python type signatures for IDE Intellisense.
"""

from collections.abc import Callable, Sequence
from typing import TypeAlias, overload

import numpy as np
import numpy.typing as npt

# Bindings reject any other dtype or a non-contiguous array with TypeError instead of copying it
_Float64Array: TypeAlias = npt.NDArray[np.float64]

class NDArray:
    def __init__(self, shape: Sequence[int], fill_value: float = 0.0) -> None: ...
    def __buffer__(self, flags: int, /) -> memoryview: ...

def add_elements(x1: _Float64Array, x2: _Float64Array, out: _Float64Array) -> None: ...
def subtract_elements(x1: _Float64Array, x2: _Float64Array, out: _Float64Array) -> None: ...
@overload
def multiply_elements(x1: _Float64Array, x2: _Float64Array, out: _Float64Array) -> None: ...
@overload
def multiply_elements(x1: _Float64Array, x2: float, out: _Float64Array) -> None: ...
def divide_elements(x1: _Float64Array, x2: _Float64Array, out: _Float64Array) -> None: ...
def compute_dot(a: _Float64Array, b: _Float64Array) -> float: ...
def apply_laplacian(padded: _Float64Array, out: _Float64Array, spacing: Sequence[float], ghost_layers: int) -> None: ...
def apply_operator(
    x: _Float64Array,
    out: _Float64Array,
    padded: _Float64Array,
    spacing: Sequence[float],
    ghost_layers: int,
    fill_ghost_layers: Callable[[_Float64Array], None],
) -> None: ...
def solve_conjugate_gradient(
    b: _Float64Array,
    x: _Float64Array,
    apply_operator: Callable[[_Float64Array, _Float64Array], None],
    compute_dot: Callable[[_Float64Array, _Float64Array], float],
    rtol: float,
    maxiter: int,
) -> None: ...
