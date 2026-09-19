from __future__ import annotations

import os
import pickle
import socket
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

import pytest

from exaflow.config import Case, InitialConditions, OutputControl, TimeControl, UniformValue
from exaflow.config.case_xml import write_case

pytestmark = pytest.mark.slow


@pytest.fixture
def run_exaflow(tmp_path: Path, build_case: Callable[..., Case]) -> Callable[..., subprocess.CompletedProcess[str]]:
    """
    Return a launcher that writes a two-step 2D case to `<tmp_path>/<name>.xml`, then runs the console entry point with the output root pointed at `<tmp_path>/runs`. Set `num_procs` above one to place that command under `mpiexec`, and pass `case` to write a case of your own instead.
    """

    def run(
        *arguments: str,
        name: str = "tiny",
        num_procs: int = 1,
        case: Case | None = None,
        stream_port: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if case is None:
            case = build_case(
                (6, 5),
                time=TimeControl(2, 0.25, 1),
                initial=InitialConditions(velocity=((UniformValue(1.0),), (UniformValue(0.0),))),
                outputs=OutputControl(checkpoint_frequency=1),
            )
        (tmp_path / f"{name}.xml").write_text(write_case(case), encoding="utf-8")
        command = [sys.executable, "-m", "exaflow.cli", *arguments]
        if num_procs > 1:
            command = ["mpiexec", "-n", str(num_procs), *command]
        environment = dict(os.environ, EXAFLOW_OUTPUT_ROOT=str(tmp_path / "runs"))
        environment.pop("EXAFLOW_STREAM_PORT", None)
        if stream_port is not None:
            environment["EXAFLOW_STREAM_PORT"] = stream_port
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            env=environment,
        )

    return run


def test_a_run_writes_one_directory_and_names_it_on_stdout(
    tmp_path: Path,
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    completed = run_exaflow("run", "--case", str(tmp_path / "tiny.xml"))

    assert completed.returncode == 0, completed.stderr[-2000:]
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    assert completed.stdout.strip() == f"Wrote {runs[0]}"
    assert "Checkpoint_Final.csv" in [entry.name for entry in runs[0].iterdir()]


def test_the_run_folder_takes_its_name_from_the_case_file(
    tmp_path: Path,
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    completed = run_exaflow("run", "--case", str(tmp_path / "lid_cavity.xml"), name="lid_cavity")

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert next((tmp_path / "runs").iterdir()).name.endswith("_lid_cavity")


def test_a_label_replaces_the_file_stem(
    tmp_path: Path,
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    completed = run_exaflow("run", "--case", str(tmp_path / "tiny.xml"), "--label", "sweep 4")

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert next((tmp_path / "runs").iterdir()).name.endswith("_sweep_4")


def test_input_xml_names_the_same_option_as_case(
    tmp_path: Path,
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    """
    Existing external commands can still pass --input-xml, but current documentation uses --case.
    """

    completed = run_exaflow("run", "--input-xml", str(tmp_path / "tiny.xml"))

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert len(list((tmp_path / "runs").iterdir())) == 1


@pytest.mark.mpi
def test_mpiexec_runs_the_standard_cli_entry_point(
    tmp_path: Path,
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    completed = run_exaflow("run", "--case", str(tmp_path / "tiny.xml"), num_procs=2)

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert len(list((tmp_path / "runs").iterdir())) == 1


def test_a_resume_continues_the_run_a_checkpoint_holds(
    tmp_path: Path,
    build_case: Callable[..., Case],
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    """
    The checkpoint carries its own case, so the second command names no XML file at all.
    """

    case = build_case(
        (6, 5),
        time=TimeControl(2, 0.25, 1),
        initial=InitialConditions(velocity=((UniformValue(1.0),), (UniformValue(0.0),))),
        outputs=OutputControl(checkpoint_frequency=1),
    )
    first = run_exaflow(
        "run",
        "--case",
        str(tmp_path / "checkpointed.xml"),
        name="checkpointed",
        case=case,
    )
    assert first.returncode == 0, first.stderr[-2000:]
    checkpoint = next((tmp_path / "runs").glob("*/Checkpoint_1.csv"))

    completed = run_exaflow("run", "--resume", str(checkpoint))

    assert completed.returncode == 0, completed.stderr[-2000:]
    continued = Path(completed.stdout.strip().removeprefix("Wrote "))
    assert (continued / "Checkpoint_2.csv").is_file()
    assert (continued / "Checkpoint_Final.csv").is_file()


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_a_stream_port_receives_the_first_and_the_last_state(
    tmp_path: Path,
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    """
    The two-step case has no intermediate stream interval, so the run sends the original and final states. Each arrives as one length-prefixed pickled grid on its own connection.
    """

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    listener.settimeout(120)
    _, port = listener.getsockname()
    steps: list[int] = []

    def accept() -> None:
        for _ in range(2):
            connection, _ = listener.accept()
            with connection, connection.makefile("rb") as stream:
                length = struct.unpack("!Q", stream.read(8))[0]
                steps.append(int(pickle.loads(stream.read(length)).field_data["StepIndex"][0]))

    thread = threading.Thread(target=accept)
    thread.start()
    try:
        completed = run_exaflow("run", "--case", str(tmp_path / "tiny.xml"), stream_port=str(port))
    finally:
        thread.join(120)
        listener.close()

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert steps == [0, 2]


def test_a_stream_port_that_is_not_a_port_is_refused(
    tmp_path: Path,
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    completed = run_exaflow("run", "--case", str(tmp_path / "tiny.xml"), stream_port="viewer")

    assert completed.returncode != 0
    assert "EXAFLOW_STREAM_PORT" in completed.stderr
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").iterdir())


def test_a_command_line_with_no_subcommand_is_refused(
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    completed = run_exaflow()

    assert completed.returncode != 0
    assert "the following arguments are required: command" in completed.stderr


def test_a_run_with_no_case_is_refused(
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    completed = run_exaflow("run")

    assert completed.returncode != 0
    assert "--case" in completed.stderr


def test_a_case_file_that_is_not_there_stops_the_run(
    tmp_path: Path,
    run_exaflow: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    completed = run_exaflow("run", "--case", str(tmp_path / "absent.xml"))

    assert completed.returncode != 0
    assert "absent.xml" in completed.stderr
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").iterdir())
