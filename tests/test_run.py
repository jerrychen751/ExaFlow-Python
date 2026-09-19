from __future__ import annotations

from pathlib import Path
from typing import Callable

from exaflow.config import Case, CheckpointFormat, OutputControl, TimeControl
from exaflow.fields import FlowState
from exaflow.run import run_case


def test_run_case_returns_the_final_local_state(build_case: Callable[..., Case]) -> None:
    case = build_case((6, 5), time=TimeControl(2, 0.25, 1))

    result = run_case(case)

    assert isinstance(result, FlowState)
    assert result.dimension == 2


def test_run_case_writes_the_standard_files(tmp_path: Path, build_case: Callable[..., Case]) -> None:
    case = build_case((6, 5), time=TimeControl(1, 0.25, 1), outputs=OutputControl(checkpoint_frequency=1))

    run_case(case, output_directory=str(tmp_path))

    assert (tmp_path / "Checkpoint_1.csv").is_file()
    assert (tmp_path / "Checkpoint_Final.csv").is_file()
    assert not list(tmp_path.glob("*.vtr"))


def test_a_vtk_case_writes_no_csv_beside_its_files(tmp_path: Path, build_case: Callable[..., Case]) -> None:
    case = build_case(
        (6, 5),
        time=TimeControl(1, 0.25, 1),
        outputs=OutputControl(checkpoint_format=CheckpointFormat.VTK, checkpoint_frequency=1),
    )

    run_case(case, output_directory=str(tmp_path))

    assert (tmp_path / "Checkpoint_1.vtr").is_file()
    assert (tmp_path / "Checkpoint_Final.vtr").is_file()
    assert not list(tmp_path.glob("*.csv"))
