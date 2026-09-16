"""
Launches one ExaFlow case under MPI and reports what it prints.
"""

from __future__ import annotations

import os
import shlex
import sys

from PySide6 import QtCore


class SimulationRunner(QtCore.QObject):
    """
    Owns the child process for one run. `start` executes `exaflow run` under MPI and returns the command for the log.

    Output arrives on `output`. `failed` reports a start fault. `finished` fires once at the end.
    """

    output = QtCore.Signal(str)
    finished = QtCore.Signal(int, str)
    failed = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._process = QtCore.QProcess(self)
        self._process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._handle_ready_read)
        self._process.errorOccurred.connect(self._handle_error)
        self._process.finished.connect(self._handle_finished)

    def is_running(self) -> bool:
        return self._process.state() != QtCore.QProcess.ProcessState.NotRunning

    def start(
        self,
        num_procs: int,
        output_root: str,
        *,
        case_path: str | None = None,
        checkpoint_path: str | None = None,
        stream_port: int | None = None,
    ) -> str:
        """
        Start a run under `mpiexec` with `num_procs` ranks, and return the command as one line of text. Give `case_path` to start a case from its input XML, `checkpoint_path` to continue the run a checkpoint holds, or both to continue that run under a replacement case. One of the two is required.

        A source run uses `python -m exaflow.cli`. A frozen run uses the bundle executable. Both processes receive `EXAFLOW_OUTPUT_ROOT`.
        """

        if case_path is None and checkpoint_path is None:
            raise ValueError("A run needs a case path, a checkpoint path, or both.")

        # The executable is the absolute path to the binary running the current process
        # For Desktop app this is the ExaFlow application (sys.frozen is True), for MPI CLI this is Python executable
        entry_arguments = (
            [sys.executable]
            if getattr(sys, "frozen", False)
            else [sys.executable, "-m", "exaflow.cli"]
        )
        launch_arguments = ["-n", str(num_procs), *entry_arguments, "run"]
        if checkpoint_path is not None:
            launch_arguments += ["--resume", os.path.abspath(checkpoint_path)]
        if case_path is not None:
            launch_arguments += ["--case", os.path.abspath(case_path)]
        display_command = shlex.join(["mpiexec", *launch_arguments])

        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("EXAFLOW_OUTPUT_ROOT", output_root)
        if stream_port is not None:
            environment.insert("EXAFLOW_STREAM_PORT", str(stream_port))
        self._process.setProcessEnvironment(environment)
        self._process.start("mpiexec", launch_arguments)
        return display_command

    def stop(self) -> None:
        """
        Kill the run. Does nothing when no process is active, so a caller need not test first.
        """

        if self.is_running():
            self._process.kill()

    def _handle_ready_read(self) -> None:
        data = bytes(self._process.readAllStandardOutput()).decode(errors="ignore")
        if data:
            self.output.emit(data.rstrip("\n"))

    def _handle_error(self, error: QtCore.QProcess.ProcessError) -> None:
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self.failed.emit(self._process.errorString())

    def _handle_finished(self, code: int, status: QtCore.QProcess.ExitStatus) -> None:
        self.finished.emit(int(code), str(status))
