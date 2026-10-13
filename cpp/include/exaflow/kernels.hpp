#pragma once

#include <cstddef>
#include <functional>
#include <span>

#include "exaflow/ndarray.hpp"

namespace exaflow {

double add(double a, double b);

double compute_dot(const NDArray& a, const NDArray& b);

void add_elements(const NDArray& x1, const NDArray& x2, NDArray& out);
void subtract_elements(const NDArray& x1, const NDArray& x2, NDArray& out);
void multiply_elements(const NDArray& x1, const NDArray& x2, NDArray& out);
void divide_elements(const NDArray& x1, const NDArray& x2, NDArray& out);
void multiply_elements(const NDArray& x1, double x2, NDArray& out);

// Reads a grid whose ghost cells have already been filled and writes only the interior Laplacian.
void apply_laplacian(
    const NDArray& padded,
    NDArray& out,
    std::span<const double> spacing,
    std::size_t ghost_layers
);

// Packs interior values into padded, refreshes homogeneous ghost cells, and writes -Laplacian(x).
void apply_operator(
    const NDArray& x,
    NDArray& out,
    NDArray& padded,
    std::span<const double> spacing,
    std::size_t ghost_layers,
    const std::function<void(NDArray&)>& fill_ghost_layers
);

// Updates x in place; callbacks supply the operator and any MPI reduction of dot products.
void solve_conjugate_gradient(
    const NDArray& b,
    NDArray& x,
    const std::function<void(const NDArray&, NDArray&)>& apply_operator,
    const std::function<double(const NDArray&, const NDArray&)>& compute_dot,
    double rtol,
    std::size_t maxiter
);

}
