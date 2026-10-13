cmake_minimum_required(VERSION 3.20)
project(exaflow_native LANGUAGES CXX)

# `uv sync` and `pip install` reach this file through scikit-build-core, which defines SKBUILD.
# A plain `cmake -S . -B build/cpp` works too, for building and testing the C++ with no Python.
option(EXAFLOW_BUILD_PYTHON "Build the exaflow._kernels Python module" ON)
option(EXAFLOW_BUILD_TESTS "Build the C++ unit tests" OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Ninja and Makefiles build one configuration and default to no optimization. Visual Studio
# and Xcode hold every configuration at once and pick one at build time with `--config`.
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT is_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 1. The kernel library: plain C++ that knows nothing about Python.
add_library(exaflow_kernels STATIC
    cpp/src/kernels.cpp
    cpp/src/dot_product.cpp
    cpp/src/elementwise.cpp
    cpp/src/laplacian.cpp
    cpp/src/pressure_operator.cpp
    cpp/src/conjugate_gradient.cpp
)
add_library(exaflow::kernels ALIAS exaflow_kernels)
target_include_directories(exaflow_kernels PUBLIC cpp/include)
if(MSVC)
    target_compile_options(exaflow_kernels PRIVATE /W4 /permissive-)
else()
    target_compile_options(exaflow_kernels PRIVATE -Wall -Wextra -Wpedantic)
endif()

# 2. The Python module: a thin pybind11 wrapper around the kernel library.
if(EXAFLOW_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_kernels MODULE cpp/bindings/module.cpp)
    target_link_libraries(_kernels PRIVATE exaflow::kernels)
    # Lands next to the Python sources of the installed package, as exaflow/_kernels.<abi>.so or .pyd.
    install(TARGETS _kernels DESTINATION exaflow)
    install(FILES cpp/stubs/exaflow/_kernels.pyi DESTINATION exaflow)
endif()

# 3. The C++ unit tests, which need network access once to fetch GoogleTest.
if(EXAFLOW_BUILD_TESTS)
    enable_testing()
    add_subdirectory(cpp/tests)
endif()
