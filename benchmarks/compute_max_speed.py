import argparse
import statistics
import time
from benchmarks.kernel_setup import build_inputs

def run_compute_max_speed(grid_size: int = 50, repeats: int = 3):

    inputs = build_inputs(n = grid_size)
    times = []

    for _ in range(repeats):
        state = inputs.state.copy()
    
        start_time = time.perf_counter()
        state.compute_max_speed()
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    median_time = statistics.median(times)
    print(f"Times taken (ms): {[round(t * 1000, 4) for t in times]}")
    print(f"Median compute_max_speed time: {median_time * 1000:.4f} ms for grid size {grid_size}^3")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    run_compute_max_speed(grid_size=arguments.grid_size, repeats=arguments.repeats)