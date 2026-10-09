"""
uv run python benchmarks/time_integrator.py [--size n] [--order k] [--warmup w] [--repeats r]
"""

from __future__ import annotations

import argparse
import statistics
import time

from mpi4py import MPI

from exaflow.numerics.operators import SpatialOperator
from exaflow.numerics.time_step import TimeIntegrator
from kernel_setup import KernelInputs, build_inputs


def time_advance(inputs: KernelInputs, order: int, warmup: int, repeats: int) -> list[float]:
    spatial = SpatialOperator(inputs.case, inputs.subdomain, MPI.COMM_WORLD)
    integrator = TimeIntegrator(spatial, order, inputs.state)
    state = inputs.state.copy()

    for _ in range(warmup):
        state = integrator.advance(state, inputs.dt)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        state = integrator.advance(state, inputs.dt)
        times.append(time.perf_counter() - start)
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description="Time TimeIntegrator.advance on an n^3 grid.")
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()

    inputs = build_inputs(arguments.size, integration_order=arguments.order)
    times = time_advance(inputs, arguments.order, arguments.warmup, arguments.repeats)

    num_cells = arguments.size**3
    median = statistics.median(times)
    print(f"{arguments.size}^3 grid, order {arguments.order}, {arguments.repeats} timed steps")
    print("steps: " + ", ".join(f"{t * 1e3:.2f} ms" for t in times))
    print(f"median: {median * 1e3:.2f} ms per step, {num_cells / median / 1e6:.1f} million cell updates per second")


if __name__ == "__main__":
    main()
