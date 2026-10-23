// Pybind builds Python modules using our C++ definitions

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
    // Register the NDArray C++ class
    // py::buffer_protocol() allows Python to read underlying data w/o copying
    py::class_<NDArray>(m, "NDArray", py::buffer_protocol())
        // Define constructor
        .def(
            py::init([](const std::vector<std::size_t>& shape, double fill_value) {
                return NDArray(shape, fill_value);
            }),
            py::arg("shape"),
            py::arg("fill_value") = 0.0
        )
        // Allow NDArray to be an object that NumPy can read from and operate on, so NDArray can be passed as input to NumPy functions
        .def_buffer([](NDArray& array) {
            const auto shape = array.get_shape();
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

    // === Bind Functions to _kernels Module ===

    // Usually pybind will try to convert Python arguments into compatible C++ types for the function
    // Because our functions take a py::array_t<> type the default behavior is copying and potentially converting the data type of the array passed in
    // We want to avoid copying, and because we also want to modify an input in-place the temporary copy made wouldn't actually change the underlying data in the argument passed in

    // ===== Element-wise Operations =====
    m.def("add_elements", &add_elements, py::arg("x1").noconvert(), py::arg("x2").noconvert(), py::arg("out").noconvert());

    m.def("subtract_elements", &subtract_elements, py::arg("x1").noconvert(), py::arg("x2").noconvert(), py::arg("out").noconvert());

    // py::overload_cast allows the multiply_elements reference resolve for overloaded C++ functions
    // because C++ functions overload based on parameter type
    m.def("multiply_elements", py::overload_cast<const NPArray&, const NPArray&, NPArray&>(&multiply_elements), py::arg("x1").noconvert(), py::arg("x2").noconvert(), py::arg("out").noconvert());

    m.def("multiply_elements", py::overload_cast<const NPArray&, double, NPArray&>(&multiply_elements), py::arg("x1").noconvert(), py::arg("x2"), py::arg("out").noconvert());

    m.def("divide_elements", &divide_elements, py::arg("x1").noconvert(), py::arg("x2").noconvert(), py::arg("out").noconvert());

    m.def("compute_dot", &compute_dot, py::arg("a").noconvert(), py::arg("b").noconvert());

    // ===== Pressure Correction Operations =====

    m.def("apply_laplacian", [](const NPArray& padded, NPArray& out, const std::vector<double>& spacing, std::size_t ghost_layers) {
        apply_laplacian(padded, out, spacing, ghost_layers);
    }, py::arg("padded").noconvert(), py::arg("out").noconvert(), py::arg("spacing"), py::arg("ghost_layers"));

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
