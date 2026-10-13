#include "exaflow/kernels.hpp"

#include <stdexcept>

namespace exaflow {

void solve_conjugate_gradient(
    const NDArray& b,
    NDArray& x,
    const std::function<void(const NDArray&, NDArray&)>& apply_operator,
    const std::function<double(const NDArray&, const NDArray&)>& compute_dot,
    double rtol,
    std::size_t maxiter
) {
    throw std::logic_error("solve_conjugate_gradient is not implemented");
}

}
