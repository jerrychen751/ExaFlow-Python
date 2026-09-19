from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path

import pytest
from PySide6 import QtCore, QtWidgets

import exaflow.gui.main_window as main_window_module
from exaflow.config.case_xml import read_case

pytestmark = pytest.mark.gui


class _FakeViewer(QtWidgets.QWidget):
    render_failed = QtCore.Signal(str)

    def clear(self) -> None:
        self.clear_calls = getattr(self, "clear_calls", 0) + 1

    def set_show_axes(self, _value: bool) -> None:
        return None

    def set_show_outline(self, _value: bool) -> None:
        return None

    def set_show_cube_axes(self, _value: bool) -> None:
        return None

    def set_show_vectors(self, _value: bool) -> None:
        return None

    def set_vector_stride(self, _value: int) -> None:
        return None

    def set_scalar_name(self, _name: str) -> None:
        return None

    def clear_held_scalar_ranges(self) -> None:
        self.held_scalar_range_clears = getattr(self, "held_scalar_range_clears", 0) + 1

    def view_pos_x(self) -> None:
        return None

    def view_neg_x(self) -> None:
        return None

    def view_pos_y(self) -> None:
        return None

    def view_neg_y(self) -> None:
        return None

    def view_pos_z(self) -> None:
        return None

    def view_neg_z(self) -> None:
        return None

    def view_iso(self) -> None:
        return None


class _FakeWatcher(QtCore.QObject):
    found = QtCore.Signal(str)

    def __init__(self, _interval: int, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)

    def set_directory(self, _directory: str) -> None:
        return None

    def set_interval(self, _interval: int) -> None:
        return None

    def set_enabled(self, _enabled: bool) -> None:
        return None

    def poll(self) -> None:
        return None

    def note_loaded(self, _path: str) -> None:
        return None


class _FakeStreamingServer(QtCore.QObject):
    data_received = QtCore.Signal(object)

    def __init__(self, port: int, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)

    def is_listening(self) -> bool:
        return True

    def read_port(self) -> int:
        return 45678


class _FakeSliceController:
    def __init__(self, _viewer: _FakeViewer, _parent: QtCore.QObject) -> None:
        self.refresh_calls = 0

    def build_toolbar(self) -> QtWidgets.QHBoxLayout:
        return QtWidgets.QHBoxLayout()

    def refresh(self) -> None:
        self.refresh_calls += 1


@pytest.fixture
def window(
    monkeypatch: pytest.MonkeyPatch,
    qt_application: QtWidgets.QApplication,
) -> Iterator[main_window_module.MainWindow]:
    monkeypatch.setattr(main_window_module, "PyVistaViewer", _FakeViewer)
    monkeypatch.setattr(main_window_module, "LatestCheckpointWatcher", _FakeWatcher)
    monkeypatch.setattr(main_window_module, "StreamingServer", _FakeStreamingServer)
    monkeypatch.setattr(main_window_module, "SliceController", _FakeSliceController)
    result = main_window_module.MainWindow()
    yield result
    result.close()


def test_the_status_beside_the_button_names_the_case(window: main_window_module.MainWindow) -> None:
    assert window._params_button.isEnabled()
    assert window._params_status.text() == "100x100x50, VTK"


def test_every_preset_loads_into_the_case_and_custom_keeps_it(window: main_window_module.MainWindow) -> None:
    labels = [window._preset_input.itemText(index) for index in range(window._preset_input.count())]
    assert labels == ["Custom", "Channel Inflow 3D", "Moving Block 3D", "Moving Patch 2D"]

    for index in range(1, window._preset_input.count()):
        window._preset_input.setCurrentIndex(index)
        window._preset_input.activated.emit(index)
        assert window._gui_case == read_case(window._preset_input.itemData(index))
        assert labels[index] in window._log_output.toPlainText()
    assert window._params_status.text() == "360x240, VTK"

    window._preset_input.setCurrentIndex(0)
    window._preset_input.activated.emit(0)
    assert window._params_status.text() == "360x240, VTK"


class _FakeRunner:
    def __init__(self) -> None:
        self.arguments: tuple[str | None, str | None, int, str, int | None] | None = None
        self.running = False
        self.stop_calls = 0

    def start(
        self,
        num_procs: int,
        output_root: str,
        *,
        case_path: str | None = None,
        checkpoint_path: str | None = None,
        stream_port: int | None = None,
    ) -> str:
        self.arguments = (case_path, checkpoint_path, num_procs, output_root, stream_port)
        self.running = True
        return "mpiexec -n 4 exaflow run --case case.xml"

    def is_running(self) -> bool:
        return self.running

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False


def test_run_writes_the_case_and_keeps_it_until_process_exit(
    tmp_path: Path,
    window: main_window_module.MainWindow,
) -> None:
    runner = _FakeRunner()
    setattr(window, "_runner", runner)
    window._mpi_processes_input.setValue(3)
    window._output_directory_input.setText(str(tmp_path / "runs"))

    window._run_simulation()

    assert runner.arguments is not None
    case_path, checkpoint_path, num_procs, output_root, stream_port = runner.arguments
    assert case_path is not None
    assert checkpoint_path is None
    assert num_procs == 3
    assert output_root == str(tmp_path / "runs")
    assert stream_port == 45678
    assert Path(case_path).is_file()
    assert read_case(case_path) == window._gui_case
    assert getattr(window._viewer, "held_scalar_range_clears") == 1

    window._handle_process_finished(0, "NormalExit")

    assert not Path(case_path).exists()


def test_clear_stops_the_run_and_resets_the_view_without_changing_the_case(
    window: main_window_module.MainWindow,
) -> None:
    runner = _FakeRunner()
    runner.running = True
    setattr(window, "_runner", runner)
    selected_case = window._gui_case
    window._run_button.setEnabled(False)
    window._resume_button.setEnabled(False)
    window._log_output.setPlainText("old output")

    window._clear_button.click()

    assert runner.stop_calls == 1
    assert getattr(window._viewer, "clear_calls") == 1
    assert getattr(window._viewer, "held_scalar_range_clears") == 1
    assert getattr(window._slice, "refresh_calls") == 1
    assert window._log_output.toPlainText() == ""
    assert window._gui_case is selected_case
    assert not window._accept_streamed_results
    assert not window._run_button.isEnabled()
    assert not window._resume_button.isEnabled()

    window._handle_process_output("late output")
    window._handle_streaming_dataset(main_window_module.pv.ImageData())
    window._handle_process_finished(1, "CrashExit")

    assert window._log_output.toPlainText() == ""
    assert window._run_button.isEnabled()
    assert window._resume_button.isEnabled()
