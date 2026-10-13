#include "exaflow/kernels.hpp"

#include <stdexcept>

namespace exaflow {

void apply_operator(
    const NDArray& x,
    NDArray& out,
    NDArray& padded,
    std::span<const double> spacing,
    std::size_t ghost_layers,
    const std::function<void(NDArray&)>& fill_ghost_layers
) {
    throw std::logic_error("apply_operator is not implemented");
}

}
