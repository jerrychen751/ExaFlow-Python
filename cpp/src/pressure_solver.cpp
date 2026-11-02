#include "exaflow/kernels.hpp"

#include <stdexcept>
#include <thread>
#include <vector>

namespace exaflow {

void apply_laplacian(
    const NPArray& padded,
    NPArray& out,
    std::span<const double> spacing,
    std::size_t ghost_layers
) {
    const std::ptrdiff_t dimension = out.ndim();
    if (dimension < 1 || dimension > 3) {
        throw std::invalid_argument("apply_laplacian needs 1, 2 or 3 axes.");
    }

    // 1D and 2D fill the last slots, because the last axis is contiguous. A missing axis has one point, no ghost layers, and a stride and inverse of 0, so its term adds nothing.
    std::ptrdiff_t sizes[3] = {1, 1, 1};
    std::ptrdiff_t pads[3] = {0, 0, 0};
    double inverses[3] = {0.0, 0.0, 0.0};
    for (std::ptrdiff_t axis = 0; axis < dimension; ++axis) {
        const std::ptrdiff_t slot = 3 - dimension + axis;
        sizes[slot] = out.shape(axis);
        pads[slot] = static_cast<std::ptrdiff_t>(ghost_layers);
        inverses[slot] = 1.0 / (spacing[axis] * spacing[axis]);
    }

    const double* input = padded.data();
    double* output = out.mutable_data();
    std::ptrdiff_t nx = sizes[0];
    std::ptrdiff_t ny = sizes[1];
    std::ptrdiff_t nz = sizes[2];
    std::ptrdiff_t pad_x = pads[0];
    std::ptrdiff_t pad_y = pads[1];
    std::ptrdiff_t pad_z = pads[2];
    double inverse_dx2 = inverses[0];
    double inverse_dy2 = inverses[1];
    double inverse_dz2 = inverses[2];

    std::ptrdiff_t Ny = ny + 2 * pad_y;
    std::ptrdiff_t Nz = nz + 2 * pad_z;
    std::ptrdiff_t stride_k = 1;
    std::ptrdiff_t stride_j = dimension >= 2 ? Nz : 0;
    std::ptrdiff_t stride_i = dimension == 3 ? Ny * Nz : 0;

    const std::ptrdiff_t num_points = nx * ny * nz;
    const std::ptrdiff_t num_threads = num_points < 200000 ? 1 : 4;

    auto sweep = [=](std::ptrdiff_t start, std::ptrdiff_t stop) {
        // Nested loops over the interior never land on a ghost point, so the innermost loop needs no check, which avoids branch misses and stays simple enough for SIMD.
        for (std::ptrdiff_t i = pad_x + start; i < pad_x + stop; ++i) {
            for (std::ptrdiff_t j = pad_y; j < pad_y + ny; ++j) {
                for (std::ptrdiff_t k = pad_z; k < pad_z + nz; ++k) {
                    std::ptrdiff_t c = i * stride_i + j * stride_j + k * stride_k;
                    std::ptrdiff_t position = (i - pad_x) * ny * nz + (j - pad_y) * nz + (k - pad_z);
                    double point = input[c];
                    double total = (input[c + stride_i] + input[c - stride_i] - 2 * point) * inverse_dx2
                                + (input[c + stride_j] + input[c - stride_j] - 2 * point) * inverse_dy2
                                + (input[c + stride_k] + input[c - stride_k] - 2 * point) * inverse_dz2;

                    output[position] = total;
                }
            }
        }
    };

    // Starting a thread costs more than sweeping a small grid, so a small grid runs here on the calling thread
    if (num_threads == 1) {
        sweep(0, nx);
        return;
    }

    std::vector<std::thread> threads; //holds the threads
    for (std::ptrdiff_t t = 0; t < num_threads; ++t) {
        std::ptrdiff_t start = nx * t / num_threads;
        std::ptrdiff_t stop = nx * (t + 1) / num_threads;

        threads.emplace_back([=] { //main thread splits the work into multi-threads.
            sweep(start, stop);
        });
    }

    for (std::thread& thread : threads) {
        thread.join();
    }
}

void apply_operator(
    const NPArray& x,
    NPArray& out,
    NPArray& padded,
    std::span<const double> spacing,
    std::size_t ghost_layers,
    const std::function<void(NPArray&)>& fill_ghost_layers
) {
    const std::ptrdiff_t dim = static_cast<std::ptrdiff_t>(out.ndim());

    const std::ptrdiff_t g = static_cast<std::ptrdiff_t>(ghost_layers);
    const double* xd = x.data();
    double* pd = padded.mutable_data();

    if (dim == 1) {
        const std::ptrdiff_t n0 = out.shape(0);
        for (std::ptrdiff_t i = 0; i < n0; ++i)
            pd[i + g] = xd[i];

    } else if (dim == 2) {
        const std::ptrdiff_t n0 = out.shape(0), n1 = out.shape(1);
        const std::ptrdiff_t s0 = padded.shape(1);
        for (std::ptrdiff_t i = 0; i < n0; ++i)
            for (std::ptrdiff_t j = 0; j < n1; ++j)
                pd[(i + g) * s0 + (j + g)] = xd[i * n1 + j];

    } else {
        const std::ptrdiff_t n0 = out.shape(0), n1 = out.shape(1), n2 = out.shape(2);
        const std::ptrdiff_t s1 = padded.shape(2);
        const std::ptrdiff_t s0 = padded.shape(1) * padded.shape(2);
        for (std::ptrdiff_t i = 0; i < n0; ++i)
            for (std::ptrdiff_t j = 0; j < n1; ++j)
                for (std::ptrdiff_t k = 0; k < n2; ++k)
                    pd[(i + g) * s0 + (j + g) * s1 + (k + g)] = xd[(i * n1 + j) * n2 + k];
    }

    fill_ghost_layers(padded);

    apply_laplacian(padded, out, spacing, ghost_layers);

    multiply_elements(out, -1.0, out);
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
