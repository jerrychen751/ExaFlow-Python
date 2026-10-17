#include "exaflow/kernels.hpp"

#include <stdexcept>

namespace exaflow {

void apply_laplacian(
    const NPArray& padded,
    NPArray& out,
    std::span<const double> spacing,
    std::size_t ghost_layers
) {
    throw std::logic_error("apply_laplacian is not implemented");
}

void apply_operator(
    const NPArray& x,
    NPArray& out,
    NPArray& padded,
    std::span<const double> spacing,
    std::size_t ghost_layers,
    const std::function<void(NPArray&)>& fill_ghost_layers
) {
    throw std::logic_error("apply_operator is not implemented");
}

void solve_conjugate_gradient(
    const NPArray& b,
    NPArray& x,
    const std::function<void(const NPArray&, NPArray&)>& apply_operator,
    const std::function<double(const NPArray&, const NPArray&)>& compute_dot,
    double rtol,
    std::size_t maxiter
) {
    throw std::logic_error("solve_conjugate_gradient is not implemented");
}

}
