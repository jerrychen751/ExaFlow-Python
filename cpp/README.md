# C++ extension

The C++ extension scaffold imports as `exaflow._kernels`. It exposes the scalar `add_c` example, and nothing in the solver calls it yet. The team kernels have C++ interfaces and implementation placeholders; calling a placeholder throws `std::logic_error` until its numerical implementation is written.

## Layout

```
CMakeLists.txt # at the repository root: what to compile and where the result goes
cpp/
    include/exaflow/ # headers: the declarations other files compile against
        kernels.hpp
        ndarray.hpp
    src/ # the definitions: plain C++ that never includes a Python header
        kernels.cpp
        dot_product.cpp
        elementwise.cpp
        laplacian.cpp
        pressure_operator.cpp
        conjugate_gradient.cpp
    bindings/
        module.cpp # the only file that includes pybind11; initializes the Python module
    stubs/exaflow/
        __init__.pyi # package-level types for local stub lookup
        _kernels.pyi # the type stub mypy reads for the compiled module
    tests/ # GoogleTest setup for the kernel library
        test_kernels.cpp
        test_ndarray.cpp
tests/test_kernels.py # the pytest module for the compiled module
```

There are three layers, and each one knows only the layer below it. The kernel library (`include/` and `src/`) is plain C++ and builds with no Python installed. The bindings expose kernels to Python and hold no numerical logic. The Python package imports `exaflow._kernels` like any other module.

The stub lives under `cpp/stubs/`. Mypy and VS Code's Pylance read it from there through the project settings. The local `__init__.pyi` preserves the package's `__version__` export. CMake installs only `_kernels.pyi` beside the compiled module in the Python package.

## Team kernel base

The shared interfaces are in `include/exaflow/kernels.hpp`. Each task has its own implementation file:

- Hemalni: `src/dot_product.cpp`, implementing `compute_dot`.
- Jerry: `src/elementwise.cpp`, implementing `add_elements`, `subtract_elements`, `multiply_elements`, and `divide_elements`.
- Miguel: `src/laplacian.cpp`, implementing `apply_laplacian`.
- Pressure operator: `src/pressure_operator.cpp`, implementing `apply_operator`.
- Conjugate gradient: `src/conjugate_gradient.cpp`, implementing `solve_conjugate_gradient`.

`NDArray` owns a row-major array of doubles with any number of dimensions, so one class holds a vector as `NDArray({n})`, a matrix as `NDArray({rows, cols})`, and a 3D field as `NDArray({nx, ny, nz})`. It supports C++23 indexing with `values[i]`, `values[row, col]` or `values[i, j, k]`, or a span of coordinates. `get_values()` returns a `std::span` view of its storage. The kernels take `NDArray` references, so each argument carries its own shape. The array must outlive the span, and assigning a new array invalidates its existing spans.

For a 3D field, construct `exaflow::NDArray field({nx, ny, nz})`. Values default to zero; a second constructor argument sets the initial value, or a braced list gives every value in storage order. `get_dimension()` returns the number of axes, `get_shape()` returns a read-only span of their lengths, and `get_size()` returns the number of values. The last axis is contiguous, so `(i, j, k)` maps to `(i * ny + j) * nz + k`. The bracket operators check the coordinate count and bounds. Zero-sized axes produce empty storage, an explicitly empty shape is rejected, and a default-constructed array is empty. The array can store more than three dimensions; the Laplacian contract remains limited to 1D–3D.

For example, once elementwise addition is implemented:

```cpp
exaflow::NDArray first({3}, {1.0, 2.0, 3.0});
exaflow::NDArray second({3}, {10.0, 20.0, 30.0});
exaflow::NDArray out({3});
exaflow::add_elements(first, second, out);
```

### Buffer contracts

- Dot product and elementwise operations use inputs of equal shape. Elementwise operations overwrite an `out` array of that shape, so callers can reuse allocations. An output may exactly alias an input; partially overlapping buffers are unsupported.
- The scalar overload of `multiply_elements` supplies the scaling needed by conjugate gradient, such as `step * direction`. It has the same output and aliasing rules.
- `compute_dot` returns the local sum of pairwise products. An MPI caller supplies the global reduction through the conjugate-gradient dot callback.
- The Laplacian uses a second-order central-difference stencil over 1, 2, or 3 axes. The shape of `out` gives the interior lengths, and `spacing` gives the positive finite cell widths in the same axis order. The last axis is contiguous. Each extent of `padded` is the matching extent of `out` plus `2 * ghost_layers`, with at least one ghost layer on each side. The caller fills those layers before calling `apply_laplacian`. `out` contains only the interior and must not overlap `padded`.
- `apply_operator` accepts interior values, an interior output buffer, and a separate reusable padded buffer. It copies the interior into the padded buffer, invokes `fill_ghost_layers`, and applies the negative Laplacian. The callback must use homogeneous pressure boundary conditions, matching `PoissonSolver._apply_operator`. The three buffers must not overlap.
- `solve_conjugate_gradient` accepts an operator callback and a dot-product callback. `b` is the right-hand side and `x` is the initial guess, which receives the solution; they have equal lengths and must not overlap. The callbacks must preserve their input values. The operator callback overwrites its equally sized output buffer and must be symmetric and positive definite on the subspace being solved. The stopping threshold is `rtol * sqrt(dot(b, b))`, with a positive finite tolerance `rtol` and a positive iteration limit `maxiter`; a zero `b` produces a zero solution. Failure to converge raises `std::runtime_error`. The caller handles pressure right-hand-side adjustments and mean removal, as the Python solver does.

These are implementation contracts; the placeholders do not yet validate inputs or perform numerical work. Boundary conditions and MPI communication stay with the caller through the callbacks. Python bindings and numerical agreement tests are added as each kernel is implemented.

## Build

`uv sync` compiles the extension through [scikit-build-core](https://scikit-build-core.readthedocs.io/), which runs CMake and places the module in the installed package. It needs a C++23 compiler:

- macOS: the Xcode Command Line Tools, from `xcode-select --install`.
- Windows: Build Tools for Visual Studio with the "Desktop development with C++" workload.

scikit-build-core downloads CMake for the build when none is on PATH.

An edit under `src/` still takes effect on the next run. An edit under `cpp/` has to be compiled, and the next `uv sync` or `uv run` does that, because `[tool.uv] cache-keys` in `pyproject.toml` lists the C++ sources.

## Test

The kernel library uses [GoogleTest](https://google.github.io/googletest/) and runs with no Python. `cpp/tests/test_ndarray.cpp` checks multidimensional storage, indexing, ownership, and invalid shapes. `cpp/tests/test_kernels.cpp` is an empty placeholder for the numerical kernels. CMake discovers tests automatically after they are added. Building the test executable needs CMake 3.20 or newer and Git on PATH, and network access the first time, to fetch GoogleTest v1.18.0:

```bash
cmake -S . -B build/cpp -DEXAFLOW_BUILD_PYTHON=OFF -DEXAFLOW_BUILD_TESTS=ON
cmake --build build/cpp --config Release
ctest --test-dir build/cpp -C Release --output-on-failure
```

The same three lines work in PowerShell. `tests/test_kernels.py` is also an empty placeholder. After adding Python tests for the extension, run:

```bash
uv run pytest tests/test_kernels.py
```

## Add a kernel

1. Declare it in a header under `cpp/include/exaflow/`.
2. Define it under `cpp/src/`. A new `.cpp` file also goes in the `add_library` list in `CMakeLists.txt`.
3. Test it in `cpp/tests/`.
4. Expose it in `cpp/bindings/module.cpp`, with `noconvert()` on every array argument.
5. Declare it in `cpp/stubs/exaflow/_kernels.pyi`.
6. Test it from Python in `tests/test_kernels.py`.

## Portability

Every change has to build and pass on macOS and on Windows. Apple clang and MSVC disagree in a few places:

- Use `std::ptrdiff_t` and `std::size_t` for indices and sizes. `ssize_t` is POSIX and MSVC does not have it.
- `long` is 64 bits on macOS and 32 bits on Windows. Use `std::int64_t` where the width matters.
- Write no bare compiler flag in CMake. `-O3`, `-march=native` and `-Wall` are GCC and Clang spellings, and MSVC rejects or ignores them. The Release build type sets optimization on every compiler; a flag that is needed goes under `if(MSVC)` with its counterpart in the `else()`.
- Keep the kernel library on the C++ standard library: `<thread>`, `<chrono>`, `<filesystem>`. No POSIX header such as `unistd.h` or `pthread.h`.
- Treat OpenMP as optional. Apple clang ships with no OpenMP runtime, and MSVC's default `/openmp` implements OpenMP 2.0, which has no `collapse` and no `simd` and takes a signed loop index only. Find it with `find_package(OpenMP)`, link it only when it is found, and keep the answers the same without it.
