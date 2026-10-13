#include "exaflow/kernels.hpp"

#include <algorithm> // compare spans
#include <stdexcept>

namespace exaflow {

// Use an unnamed namespace to contain private helper function local to this file
namespace {
bool is_same_shape(const NDArray& a, const NDArray& b) {
    return std::ranges::equal(a.get_shape(), b.get_shape());
}
}

void add_elements(const NDArray& x1, const NDArray& x2, NDArray& out) {
    if (!is_same_shape(x1, x2) || !is_same_shape(x2, out)) {
        throw std::invalid_argument("Parameters don't have the same shape");
    }

    auto v1 = x1.get_values();
    auto v2 = x2.get_values();
    auto res = out.get_values();
    for (std::size_t i = 0; i < x1.get_size(); ++i) {
        res[i] = v1[i] + v2[i];
    }
}

void subtract_elements(const NDArray& x1, const NDArray& x2, NDArray& out) {
    if (!is_same_shape(x1, x2) || !is_same_shape(x2, out)) {
        throw std::invalid_argument("Parameters don't have the same shape");
    }

    auto v1 = x1.get_values();
    auto v2 = x2.get_values();
    auto res = out.get_values();
    for (std::size_t i = 0; i < x1.get_size(); ++i) {
        res[i] = v1[i] - v2[i];
    }
}

void multiply_elements(const NDArray& x1, const NDArray& x2, NDArray& out) {
    if (!is_same_shape(x1, x2) || !is_same_shape(x2, out)) {
        throw std::invalid_argument("Parameters don't have the same shape");
    }

    auto v1 = x1.get_values();
    auto v2 = x2.get_values();
    auto res = out.get_values();
    for (std::size_t i = 0; i < x1.get_size(); ++i) {
        res[i] = v1[i] * v2[i];
    }
}

void multiply_elements(const NDArray& x1, double x2, NDArray& out) {
    if (!is_same_shape(x1, out)) {
        throw std::invalid_argument("Parameters don't have the same shape");
    }

    auto v1 = x1.get_values();
    auto res = out.get_values();
    for (std::size_t i = 0; i < x1.get_size(); ++i) {
        res[i] = v1[i] * x2;
    }
}

void divide_elements(const NDArray& x1, const NDArray& x2, NDArray& out) {
    if (!is_same_shape(x1, x2) || !is_same_shape(x2, out)) {
        throw std::invalid_argument("Parameters don't have the same shape");
    }

    auto v1 = x1.get_values();
    auto v2 = x2.get_values();
    auto res = out.get_values();
    for (std::size_t i = 0; i < x1.get_size(); ++i) {
        // Since these are floats, division by zero is actually not undefined
        // inf if numerator is positive, -inf if numerator is negative, and nan for 0.0 / 0.0
        res[i] = v1[i] / v2[i];
    }
}

}
