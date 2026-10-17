#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <vector>

#include "exaflow/kernels.hpp"
#include "exaflow/ndarray.hpp"

namespace py = pybind11;
using namespace exaflow;

PYBIND11_MODULE(_kernels, m) {
    m.def("add_c", &add, py::arg("a"), py::arg("b")); // define name of function and args for Python

    // NDArray owns its storage, so Python allocates through it and np.asarray views that storage without a copy
    py::class_<NDArray>(m, "NDArray", py::buffer_protocol())
        // pybind11 has no caster for std::span, so the shape arrives as a vector
        .def(
            py::init([](const std::vector<std::size_t>& shape, double fill_value) {
                return NDArray(shape, fill_value);
            }),
            py::arg("shape"),
            py::arg("fill_value") = 0.0
        )
        .def_buffer([](NDArray& array) {
            const auto shape = array.get_shape();
            // Strides are in bytes; the last axis is contiguous. The constructor rejects an empty shape, so the loop never underflows.
            std::vector<py::ssize_t> strides(shape.size(), sizeof(double));
            for (std::size_t axis = shape.size() - 1; axis > 0; --axis) {
                strides[axis - 1] = strides[axis] * static_cast<py::ssize_t>(shape[axis]);
            }
            return py::buffer_info(
                array.get_values().data(),
                sizeof(double),
                py::format_descriptor<double>::format(),
                static_cast<py::ssize_t>(shape.size()),
                std::vector<py::ssize_t>(shape.begin(), shape.end()),
                strides
            );
        });

    // Element-wise operation functions
    m.def("add_elements", &add_elements, py::arg("x1").noconvert(), py::arg("x2").noconvert(), py::arg("out").noconvert());
    m.def("subtract_elements", &subtract_elements, py::arg("x1").noconvert(), py::arg("x2").noconvert(), py::arg("out").noconvert());
    m.def("multiply_elements", py::overload_cast<const NPArray&, const NPArray&, NPArray&>(&multiply_elements), py::arg("x1").noconvert(), py::arg("x2").noconvert(), py::arg("out").noconvert());
    // x2 keeps conversion on so a Python int is accepted as the scalar
    m.def("multiply_elements", py::overload_cast<const NPArray&, double, NPArray&>(&multiply_elements), py::arg("x1").noconvert(), py::arg("x2"), py::arg("out").noconvert());
    m.def("divide_elements", &divide_elements, py::arg("x1").noconvert(), py::arg("x2").noconvert(), py::arg("out").noconvert());

    m.def("compute_dot", &compute_dot, py::arg("a").noconvert(), py::arg("b").noconvert());

    // pybind11 has no caster for std::span, so spacing arrives as a vector
    m.def("apply_laplacian", [](const NPArray& padded, NPArray& out, const std::vector<double>& spacing, std::size_t ghost_layers) {
        apply_laplacian(padded, out, spacing, ghost_layers);
    }, py::arg("padded").noconvert(), py::arg("out").noconvert(), py::arg("spacing"), py::arg("ghost_layers"));
    // pybind11/functional.h wraps the Python callback in a std::function, which receives the caller's own padded array
    m.def("apply_operator", [](const NPArray& x, NPArray& out, NPArray& padded, const std::vector<double>& spacing, std::size_t ghost_layers, const std::function<void(NPArray&)>& fill_ghost_layers) {
        apply_operator(x, out, padded, spacing, ghost_layers, fill_ghost_layers);
    }, py::arg("x").noconvert(), py::arg("out").noconvert(), py::arg("padded").noconvert(), py::arg("spacing"), py::arg("ghost_layers"), py::arg("fill_ghost_layers"));
    m.def("solve_conjugate_gradient", &solve_conjugate_gradient, py::arg("b").noconvert(), py::arg("x").noconvert(), py::arg("apply_operator"), py::arg("compute_dot"), py::arg("rtol"), py::arg("maxiter"));
}

// Hemalni: dot product
// Jerry: elementwise add/subtract/mult/divide
// Miguel: laplacian operator
// _apply_operator
// conjugate gradient
