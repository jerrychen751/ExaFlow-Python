"""
uv run python benchmarks/whole_simulation.py <case> [--repeats n]
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

from mpi4py import MPI

from exaflow.config import Case
from exaflow.config.case_xml import read_case
from exaflow.session import SimulationSession


def time_one_run(case: Case, comm: MPI.Intracomm) -> tuple[float, int]:
    session = SimulationSession(case, comm)
    comm.Barrier()
    start = time.perf_counter()
    session.run_until_complete(write_initial=False)
    elapsed = time.perf_counter() - start
    return float(comm.allreduce(elapsed, op=MPI.MAX)), session.step_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Time whole ExaFlow runs of one benchmark case.")
    parser.add_argument("case", help="Path to a case XML file, such as benchmarks/cases/small_50x50x25.xml.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed runs; the median is reported.")
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error(f"--repeats must be at least 1, got {arguments.repeats}.")

    comm = MPI.COMM_WORLD
    case = read_case(arguments.case)
    walls = []
    num_steps = 0
    for _ in range(arguments.repeats):
        wall, num_steps = time_one_run(case, comm)
        walls.append(wall)

    if comm.Get_rank() != 0:
        return
    num_cells = math.prod(case.grid.shape)
    median = statistics.median(walls)
    print(f"{arguments.case}: {num_cells} cells, {num_steps} steps, {comm.Get_size()} rank(s)")
    print("runs: " + ", ".join(f"{wall:.3f} s" for wall in walls))
    print(
        f"median: {median:.3f} s, {median / num_steps * 1e3:.2f} ms per step, "
        f"{num_cells * num_steps / median / 1e6:.1f} million cell updates per second"
    )


if __name__ == "__main__":
    main()
