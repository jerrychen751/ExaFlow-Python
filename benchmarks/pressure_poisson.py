import argparse
import statistics
import time

from mpi4py import MPI

from exaflow.numerics.pressure_poisson import PoissonSolver
from kernel_setup import build_inputs, KernelInputs

def time_one_projection(inputs: KernelInputs) -> float:
    kernel = PoissonSolver(inputs.case, inputs.subdomain, MPI.COMM_WORLD)
    state = inputs.state.copy()
    start = time.perf_counter()
    kernel.project(state, inputs.dt)
    return time.perf_counter() - start

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=50) # n^3 grid size
    parser.add_argument("--repeats", type=int, default=3) # number of times the projection operation is repeated
    arguments = parser.parse_args()

    inputs = build_inputs(arguments.size)
    times = [round(time_one_projection(inputs), 2) for _ in range(arguments.repeats)]
    median = statistics.median(times)
    print(f"Times taken (s): {times}")
    print(f"Median time taken (s): {median}")

if __name__ == "__main__":
    main()