from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from PySide6 import QtCore, QtWidgets

from exaflow.gui.simulation_runner import SimulationRunner

pytestmark = pytest.mark.gui


class FakeProcess:
    def __init__(self) -> None:
        self.environment: QtCore.QProcessEnvironment | None = None
        self.program = ""
        self.arguments: list[str] = []

    def setProcessEnvironment(self, environment: QtCore.QProcessEnvironment) -> None:
        self.environment = environment

    def start(self, program: str, arguments: list[str]) -> None:
        self.program = program
        self.arguments = arguments


def test_source_run_uses_the_standard_cli_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qt_application: QtWidgets.QApplication,
) -> None:
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    runner = SimulationRunner()
    process = FakeProcess()
    setattr(runner, "_process", process)
    case_path = tmp_path / "case with spaces.xml"

    command = runner.start(4, str(tmp_path / "runs"), case_path=str(case_path))

    assert process.program == "mpiexec"
    assert process.arguments == [
        "-n",
        "4",
        sys.executable,
        "-m",
        "exaflow.cli",
        "run",
        "--case",
        os.path.abspath(case_path),
    ]
    assert process.environment is not None
    assert process.environment.value("EXAFLOW_OUTPUT_ROOT") == str(tmp_path / "runs")
    assert not process.environment.contains("EXAFLOW_STREAM_PORT")
    assert "exaflow.cli run --case" in command


def test_a_stream_port_reaches_the_run_through_the_environment(
    tmp_path: Path,
    qt_application: QtWidgets.QApplication,
) -> None:
    runner = SimulationRunner()
    process = FakeProcess()
    setattr(runner, "_process", process)

    runner.start(1, str(tmp_path / "runs"), case_path=str(tmp_path / "case.xml"), stream_port=54321)

    assert process.environment is not None
    assert process.environment.value("EXAFLOW_STREAM_PORT") == "54321"


def test_frozen_run_restarts_the_bundle_in_cli_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qt_application: QtWidgets.QApplication,
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    runner = SimulationRunner()
    process = FakeProcess()
    setattr(runner, "_process", process)
    case_path = tmp_path / "case.xml"

    runner.start(2, str(tmp_path / "runs"), case_path=str(case_path))

    assert process.arguments == [
        "-n",
        "2",
        sys.executable,
        "run",
        "--case",
        str(case_path),
    ]
