#include <pybind11/pybind11.h>

#include "exaflow/kernels.hpp"

namespace py = pybind11;


PYBIND11_MODULE(_kernels, m) {
    m.def("add_c", &exaflow::add, py::arg("a"), py::arg("b")); // define name of function and args for Python
}

// Hemalni: dot product
// Jerry: elementwise add/subtract/mult/divide
// Miguel: laplacian operator
// _apply_operator
// conjugate gradient