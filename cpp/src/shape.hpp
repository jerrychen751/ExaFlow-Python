// Not placed in cpp/include/exaflow/ because CMakeLists.txt makes cpp/include public
// Only files in cpp/src/ use it

#include "exaflow/kernels.hpp"

namespace exaflow {

inline bool is_same_shape(const NPArray& a, const NPArray& b) {
    // inline ensures that when this file is included multiple times there's no duplicate function definition

    // std::span(Pointer, Size)
    // std::ranges::equal() checks sizes of spans first, and then iterates through elementwise to compare
    return std::ranges::equal(std::span(a.shape(), a.ndim()), std::span(b.shape(), b.ndim()));
}

}