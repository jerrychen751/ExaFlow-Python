import argparse
import statistics
import time
from benchmarks.kernel_setup import build_inputs

def run_set_sum(grid_size: int = 50, repeats: int = 3):
    
    inputs = build_inputs(n = grid_size)
    rate = inputs.rate
    factor = inputs.dt
    times = []

    for _ in range(repeats):
        base = inputs.state.copy()
        state = inputs.state.copy()

        start_time = time.perf_counter()
        state.set_sum(base, rate, factor)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    median_time = statistics.median(times)
    print(f"Times taken (ms): {[round(t * 1000, 4) for t in times]}")
    print(f"Median set_sum time: {median_time * 1000:.4f} ms for grid size {grid_size}^3")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    run_set_sum(grid_size=arguments.grid_size, repeats=arguments.repeats)