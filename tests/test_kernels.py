from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from exaflow import _kernels
from exaflow._kernels import NDArray


def test_ndarray_view_shares_storage_with_numpy() -> None:
    array = NDArray((2, 3), 1.5)
    view = np.asarray(array)

    assert view.shape == (2, 3)
    assert view.dtype == np.float64
    assert view.flags.c_contiguous
    assert np.all(view == 1.5)

    # A write through one view must be visible through a second one, which proves neither is a copy
    view[1, 2] = 7.0
    assert np.asarray(array)[1, 2] == 7.0


def test_ndarray_view_keeps_storage_alive_after_the_array_name_is_gone() -> None:
    view = np.asarray(NDArray((4,), 2.0))

    assert np.all(view == 2.0)


def test_ndarray_rejects_an_empty_shape() -> None:
    with pytest.raises(ValueError):
        NDArray(())


def test_ndarray_with_a_zero_extent_gives_an_empty_view() -> None:
    view = np.asarray(NDArray((0, 3)))

    assert view.shape == (0, 3)


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        (_kernels.add_elements, np.add),
        (_kernels.subtract_elements, np.subtract),
        (_kernels.multiply_elements, np.multiply),
        (_kernels.divide_elements, np.divide),
    ],
)
def test_elementwise_kernel_matches_numpy(
    kernel: Callable[[np.ndarray, np.ndarray, np.ndarray], None],
    expected: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    rng = np.random.default_rng(0)
    first = rng.uniform(-2.0, 2.0, (3, 4, 5))
    second = rng.uniform(0.5, 2.0, (3, 4, 5))  # kept away from zero so divide stays finite
    out = np.empty_like(first)

    kernel(first, second, out)

    assert np.array_equal(out, expected(first, second))


@pytest.mark.parametrize("scalar", [2.5, 3])
def test_scalar_multiply_matches_numpy(scalar: float) -> None:
    values = np.arange(6.0).reshape(2, 3)
    out = np.empty_like(values)

    _kernels.multiply_elements(values, scalar, out)

    assert np.array_equal(out, values * scalar)


def test_elementwise_kernel_can_write_into_one_of_its_inputs() -> None:
    first = np.array([1.0, 2.0, 3.0])
    second = np.array([10.0, 20.0, 30.0])

    _kernels.add_elements(first, second, first)

    assert np.array_equal(first, [11.0, 22.0, 33.0])


def test_elementwise_kernel_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        _kernels.add_elements(np.ones(3), np.ones(4), np.empty(3))


# Converting either array would hand the kernel a temporary copy, so a write to out would never reach the caller
@pytest.mark.parametrize(
    "out",
    [np.empty(3, dtype=np.float32), np.empty(6)[::2]],
    ids=["float32", "strided"],
)
def test_elementwise_kernel_rejects_an_array_it_would_have_to_copy(out: np.ndarray) -> None:
    with pytest.raises(TypeError):
        _kernels.add_elements(np.ones(3), np.ones(3), out)


def test_elementwise_kernel_rejects_a_read_only_out() -> None:
    out = np.empty(3)
    out.flags.writeable = False

    with pytest.raises(ValueError):
        _kernels.add_elements(np.ones(3), np.ones(3), out)


# Each placeholder raises std::logic_error, which reaches Python as RuntimeError only after every argument, callbacks included, has converted
def test_placeholder_kernels_accept_numpy_arguments() -> None:
    padded = np.zeros((6, 6))
    out = np.zeros((4, 4))

    with pytest.raises(RuntimeError, match="apply_laplacian is not implemented"):
        _kernels.apply_laplacian(padded, out, [0.1, 0.1], 1)

    with pytest.raises(RuntimeError, match="apply_operator is not implemented"):
        _kernels.apply_operator(np.zeros((4, 4)), out, padded, [0.1, 0.1], 1, lambda array: None)

    with pytest.raises(RuntimeError, match="solve_conjugate_gradient is not implemented"):
        _kernels.solve_conjugate_gradient(
            np.ones(4),
            np.zeros(4),
            lambda x, result: None,
            lambda a, b: float(np.dot(a, b)),
            1e-8,
            100,
        )
