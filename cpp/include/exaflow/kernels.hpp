#pragma once

#include <cstddef>
#include <functional>
#include <span>

#include <pybind11/numpy.h>

namespace exaflow {

// Checks whether Python object is a numpy array, float64 type, and in row-major order
using NPArray = pybind11::array_t<double, pybind11::array::c_style>;

double compute_dot(const NPArray& a, const NPArray& b);

void add_elements(const NPArray& x1, const NPArray& x2, NPArray& out);
void subtract_elements(const NPArray& x1, const NPArray& x2, NPArray& out);
void multiply_elements(const NPArray& x1, const NPArray& x2, NPArray& out);
void divide_elements(const NPArray& x1, const NPArray& x2, NPArray& out);
void multiply_elements(const NPArray& x1, double x2, NPArray& out);

// Reads a grid whose ghost cells have already been filled and writes only the interior Laplacian.
void apply_laplacian(
    const NPArray& padded,
    NPArray& out,
    std::span<const double> spacing,
    std::size_t ghost_layers
);

// Packs interior values into padded, refreshes homogeneous ghost cells, and writes -Laplacian(x).
void apply_operator(
    const NPArray& x,
    NPArray& out,
    NPArray& padded,
    std::span<const double> spacing,
    std::size_t ghost_layers,
    const std::function<void(NPArray&)>& fill_ghost_layers
);

// Updates x in place; callbacks supply the operator and any MPI reduction of dot products.
void solve_conjugate_gradient(
    const NPArray& b,
    NPArray& x,
    const std::function<void(const NPArray&, NPArray&)>& apply_operator,
    const std::function<double(const NPArray&, const NPArray&)>& compute_dot,
    double rtol,
    std::size_t maxiter
);

}
