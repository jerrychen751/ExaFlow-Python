#include "exaflow/kernels.hpp"
#include "shape.hpp" // searches same folder for file

#include <stdexcept>
#include <numeric>

namespace exaflow {

double compute_dot(const NPArray& a, const NPArray& b) {
    if (!is_same_shape(a, b)) {
        throw std::invalid_argument("Input shapes do not match");
    }

    // Initial value should be 0.0 instead of 0 to avoid int in summation
    return std::inner_product(a.data(), a.data() + a.size(), b.data(), 0.0);
}

void add_elements(const NPArray& x1, const NPArray& x2, NPArray& out) {
    if (!is_same_shape(x1, x2) || !is_same_shape(x2, out)) {
        throw std::invalid_argument("Parameters don't have the same shape");
    }

    auto v1 = x1.data();
    auto v2 = x2.data();
    auto res = out.mutable_data();
    for (pybind11::ssize_t i = 0; i < x1.size(); ++i) {
        res[i] = v1[i] + v2[i];
    }
}

void subtract_elements(const NPArray& x1, const NPArray& x2, NPArray& out) {
    if (!is_same_shape(x1, x2) || !is_same_shape(x2, out)) {
        throw std::invalid_argument("Parameters don't have the same shape");
    }

    auto v1 = x1.data();
    auto v2 = x2.data();
    auto res = out.mutable_data();
    for (pybind11::ssize_t i = 0; i < x1.size(); ++i) {
        res[i] = v1[i] - v2[i];
    }
}

void multiply_elements(const NPArray& x1, const NPArray& x2, NPArray& out) {
    if (!is_same_shape(x1, x2) || !is_same_shape(x2, out)) {
        throw std::invalid_argument("Parameters don't have the same shape");
    }

    auto v1 = x1.data();
    auto v2 = x2.data();
    auto res = out.mutable_data();
    for (pybind11::ssize_t i = 0; i < x1.size(); ++i) {
        res[i] = v1[i] * v2[i];
    }
}

void multiply_elements(const NPArray& x1, double x2, NPArray& out) {
    if (!is_same_shape(x1, out)) {
        throw std::invalid_argument("Parameters don't have the same shape");
    }

    auto v1 = x1.data();
    auto res = out.mutable_data();
    for (pybind11::ssize_t i = 0; i < x1.size(); ++i) {
        res[i] = v1[i] * x2;
    }
}

void divide_elements(const NPArray& x1, const NPArray& x2, NPArray& out) {
    if (!is_same_shape(x1, x2) || !is_same_shape(x2, out)) {
        throw std::invalid_argument("Parameters don't have the same shape");
    }

    auto v1 = x1.data();
    auto v2 = x2.data();
    auto res = out.mutable_data();
    for (pybind11::ssize_t i = 0; i < x1.size(); ++i) {
        // Since these are floats, division by zero is actually not undefined
        // inf with the sign of numerator times denominator, and nan for 0.0 / 0.0
        res[i] = v1[i] / v2[i];
    }
}

}
