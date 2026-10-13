#include "exaflow/kernels.hpp"

#include <stdexcept>

namespace exaflow {

void apply_laplacian(
    const NDArray& padded,
    NDArray& out,
    std::span<const double> spacing,
    std::size_t ghost_layers
) {
    throw std::logic_error("apply_laplacian is not implemented");
}

}
