"""
The `exaflow` console entry point.

    exaflow run --case examples/input_template.xml
    mpiexec -n 4 exaflow run --case examples/input_template.xml
    mpiexec -n 4 exaflow run --resume ~/Documents/ExaFlow/2026-08-31_120000_template/Checkpoint_400.npz

This is the supported way to run a case without writing a driver, and it is what the GUI launches.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import cast

from .config.case_xml import read_case
from .io.storage import create_run_directory
from .run import resume_case, run_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="exaflow", description="Run an ExaFlow case.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run a case given as an input XML file.")
    run_parser.add_argument(
        "--case",
        "--input-xml",
        dest="case",
        default=None,
        help="Path to the input XML file.",
    )
    run_parser.add_argument(
        "--resume",
        dest="resume",
        default=None,
        help="Path to a checkpoint to continue from. Give --case beside it to replace the stored case.",
    )
    run_parser.add_argument(
        "--label",
        default=None,
        help="Name for the run folder; defaults to the file stem.",
    )
    arguments, _ = parser.parse_known_args(argv)
    if arguments.case is None and arguments.resume is None:
        parser.error("run needs --case or --resume.")
    stream_port_text = os.environ.get("EXAFLOW_STREAM_PORT", "").strip()
    stream_port = None
    if stream_port_text:
        stream_port = int(stream_port_text) if stream_port_text.isdecimal() else 0
        if not 1 <= stream_port <= 65535:
            parser.error(f"EXAFLOW_STREAM_PORT must be a port number from 1 to 65535, got {stream_port_text!r}.")

    try:
        import mpi4py.MPI as mpi
    except ImportError:
        comm = None
    else:
        comm = mpi.COMM_WORLD
    case = read_case(arguments.case) if arguments.case is not None else None
    label = arguments.label or Path(cast(str, arguments.resume or arguments.case)).stem
    run_directory = create_run_directory(label, comm)

    if arguments.resume is None:
        assert case is not None, "The parser refuses a run that names neither --case nor --resume."
        run_case(case, comm, output_directory=run_directory, stream_port=stream_port)
    else:
        resume_case(arguments.resume, comm, case=case, output_directory=run_directory, stream_port=stream_port)
    rank = int(comm.Get_rank()) if comm is not None else 0
    if rank == 0:
        print(f"Wrote {run_directory}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
