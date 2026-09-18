from __future__ import annotations

import os
import tempfile
import traceback
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import pyvista as pv
from PySide6 import QtCore, QtWidgets

from ..config import Case
from ..config.case_xml import read_case, write_case
from ..io.storage import resolve_output_root
from .help_dialog import HelpDialog
from .result_watcher import LatestResultWatcher
from .settings_dialog import SessionSettings, SettingsDialog
from .sim_parameters_dialog import SimulationParametersDialog, build_default_case
from .simulation_runner import SimulationRunner
from .slice_controller import SliceController
from .streaming import DEFAULT_STREAMING_PORT, StreamingServer
from .viewer import PyVistaViewer


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ExaFlow - Python GUI")

        # Process used to run simulations
        # Register handlers so that the main window can react to a child process
        self._runner = SimulationRunner(self)
        self._runner.output.connect(self._append_log)
        self._runner.finished.connect(self._handle_process_finished)
        self._runner.failed.connect(self._handle_process_failed)

        # State
        self._gui_case: Case = build_default_case()
        self._run_case_directory: tempfile.TemporaryDirectory[str] | None = None # a scratch directory holds XML case
        self._session_settings = SessionSettings()

        # UI
        self._build_ui()

        # Directory watcher
        self._watcher = LatestResultWatcher(self._session_settings.autoload_interval_ms, self)
        self._watcher.found.connect(self._load_file_path)
        self._refresh_watcher()

        # Streaming server for real-time updates
        self._streaming_server: StreamingServer | None = None
        try:
            self._streaming_server = StreamingServer(
                port=DEFAULT_STREAMING_PORT,
                parent=self,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Streaming Error",
                f"Unable to start streaming server:\n{exc}",
            )
        else:
            if not self._streaming_server.is_listening():
                QtWidgets.QMessageBox.critical(
                    self,
                    "Streaming Error",
                    "Unable to start streaming server.",
                )
                self._streaming_server = None
            else:
                # Signal is emitted only after full message is read and successfully unpickled (see server.py)
                self._streaming_server.data_received.connect(self._handle_streaming_dataset)

    # ------------------------- UI -------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)

        # Left controls
        controls = QtWidgets.QWidget(central)
        controls_layout = QtWidgets.QVBoxLayout(controls)
        form = QtWidgets.QFormLayout()

        self._params_button = QtWidgets.QPushButton("Simulation Params…", controls)
        self._params_button.clicked.connect(self._open_simulation_params_dialog)
        self._params_status = QtWidgets.QLabel(controls)
        self._refresh_case_status()
        params_row = QtWidgets.QHBoxLayout()
        params_row.addWidget(self._params_button)
        params_row.addWidget(self._params_status, 1)
        form.addRow("Case", params_row)

        self._preset_input = QtWidgets.QComboBox(controls)
        self._preset_input.addItem("Custom")
        for preset_path in sorted(Path(str(files("exaflow.gui").joinpath("presets"))).glob("*.xml")):
            self._preset_input.addItem(preset_path.stem.replace("_", " ").title(), str(preset_path))
        self._preset_input.activated.connect(self._load_preset)
        form.addRow("Preset", self._preset_input)

        # MPI processes
        self._mpi_processes_input = QtWidgets.QSpinBox(controls)
        self._mpi_processes_input.setRange(1, 512)
        self._mpi_processes_input.setValue(4)
        form.addRow("MPI processes", self._mpi_processes_input)

        # Output directory (for watcher)
        self._output_directory_input = QtWidgets.QLineEdit(controls)
        self._output_directory_input.setText(resolve_output_root())
        self._output_directory_input.editingFinished.connect(self._refresh_watcher)
        self._output_directory_browse_button = QtWidgets.QPushButton("Browse…", controls)
        self._output_directory_browse_button.clicked.connect(self._browse_output_directory)
        output_directory_row = QtWidgets.QHBoxLayout()
        output_directory_row.addWidget(self._output_directory_input)
        output_directory_row.addWidget(self._output_directory_browse_button)
        form.addRow("Output root", output_directory_row)

        # Auto-load latest
        controls_layout.addLayout(form)

        # Run controls
        run_controls_row = QtWidgets.QHBoxLayout()
        self._run_button = QtWidgets.QPushButton("Run", controls)
        self._run_button.clicked.connect(self._run_simulation)
        self._resume_button = QtWidgets.QPushButton("Resume…", controls)
        self._resume_button.clicked.connect(self._resume_simulation)
        self._stop_button = QtWidgets.QPushButton("Stop", controls)
        self._stop_button.clicked.connect(self._stop_simulation)
        self._open_file_button = QtWidgets.QPushButton("Open File…", controls)
        self._open_file_button.clicked.connect(self._open_file)
        self._help_button = QtWidgets.QPushButton("?", controls)
        self._help_button.clicked.connect(self._show_help)
        self._help_button.setMaximumWidth(30)
        self._settings_button = QtWidgets.QToolButton(controls)
        self._settings_button.setToolTip("Session settings")
        settings_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView)
        self._settings_button.setIcon(settings_icon)
        self._settings_button.clicked.connect(self._open_settings_dialog)
        run_controls_row.addWidget(self._run_button)
        run_controls_row.addWidget(self._resume_button)
        run_controls_row.addWidget(self._stop_button)
        run_controls_row.addWidget(self._open_file_button)
        run_controls_row.addWidget(self._help_button)
        run_controls_row.addWidget(self._settings_button)
        controls_layout.addLayout(run_controls_row)

        # Log output
        self._log_output = QtWidgets.QPlainTextEdit(controls)
        self._log_output.setReadOnly(True)
        self._log_output.setMaximumBlockCount(5000)
        controls_layout.addWidget(QtWidgets.QLabel("Run log"))
        controls_layout.addWidget(self._log_output, 1)

        # Right viewer
        self._viewer = PyVistaViewer(central)
        self._viewer.render_failed.connect(self._append_log)

        # Build a right-side layout with a small toolbar for viewer controls
        right = QtWidgets.QWidget(central)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar row: overlays and vectors
        toolbar = QtWidgets.QHBoxLayout()

        self._axes_checkbox = QtWidgets.QCheckBox("Axes")
        self._axes_checkbox.setChecked(True)
        self._axes_checkbox.toggled.connect(lambda v: self._viewer.set_show_axes(bool(v)))
        toolbar.addWidget(self._axes_checkbox)

        self._outline_checkbox = QtWidgets.QCheckBox("Outline")
        self._outline_checkbox.setChecked(True)
        self._outline_checkbox.toggled.connect(lambda v: self._viewer.set_show_outline(bool(v)))
        toolbar.addWidget(self._outline_checkbox)

        self._cube_axes_checkbox = QtWidgets.QCheckBox("Cube Axes")
        self._cube_axes_checkbox.setChecked(True)
        self._cube_axes_checkbox.toggled.connect(lambda v: self._viewer.set_show_cube_axes(bool(v)))
        toolbar.addWidget(self._cube_axes_checkbox)

        self._vectors_checkbox = QtWidgets.QCheckBox("Vectors")
        self._vectors_checkbox.setChecked(False)
        self._vectors_checkbox.toggled.connect(lambda v: self._viewer.set_show_vectors(bool(v)))
        toolbar.addWidget(self._vectors_checkbox)

        toolbar.addWidget(QtWidgets.QLabel("Stride"))
        self._vector_stride_spinbox = QtWidgets.QSpinBox()
        self._vector_stride_spinbox.setRange(1, 50)
        self._vector_stride_spinbox.setValue(4)
        self._vector_stride_spinbox.valueChanged.connect(lambda n: self._viewer.set_vector_stride(int(n)))
        toolbar.addWidget(self._vector_stride_spinbox)

        toolbar.addStretch(1)

        # Camera preset buttons
        camera_buttons = [
            ("+X", "view_pos_x"), ("-X", "view_neg_x"),
            ("+Y", "view_pos_y"), ("-Y", "view_neg_y"),
            ("+Z", "view_pos_z"), ("-Z", "view_neg_z"),
            ("ISO", "view_iso"),
        ]
        for text, method_name in camera_buttons:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(getattr(self._viewer, method_name))
            toolbar.addWidget(button)

        self._slice = SliceController(self._viewer, self)

        right_layout.addLayout(toolbar)
        right_layout.addLayout(self._slice.build_toolbar())
        right_layout.addWidget(self._viewer)

        # Splitter layout
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, central)
        controls.setMinimumWidth(360)
        splitter.addWidget(controls)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

    # ------------------------- Process handling -------------------------
    def _run_simulation(self) -> None:
        mpi_processes = int(self._mpi_processes_input.value())
        self._clear_run_case_directory()
        self._run_case_directory = tempfile.TemporaryDirectory(prefix="exaflow-gui-")
        case_path = Path(self._run_case_directory.name) / "gui_case.xml"
        try:
            case_path.write_text(write_case(self._gui_case), encoding="utf-8")
        except OSError as exc:
            self._clear_run_case_directory()
            QtWidgets.QMessageBox.critical(self, "Failed to write the case", str(exc))
            return

        self._log_output.clear()
        self._run_button.setEnabled(False)
        self._resume_button.setEnabled(False)
        display_command = self._runner.start(
            mpi_processes,
            self._output_directory_input.text().strip() or resolve_output_root(),
            case_path=str(case_path),
            stream_port=self._read_stream_port(),
        )
        self._append_log(f"[{self._format_time()}] Case saved to {case_path}")
        self._append_log(f"[{self._format_time()}] Starting: {display_command}")

    def _resume_simulation(self) -> None:
        """
        Continue the run one checkpoint holds. The case comes from the checkpoint itself, not from the form, because the fields it stores belong to the case that produced them.
        """

        checkpoint_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Resume from a checkpoint",
            self._output_directory_input.text().strip() or os.getcwd(),
            "ExaFlow checkpoint (Checkpoint_*.npz);;All files (*)",
        )
        if not checkpoint_path:
            return

        self._log_output.clear()
        self._run_button.setEnabled(False)
        self._resume_button.setEnabled(False)
        display_command = self._runner.start(
            int(self._mpi_processes_input.value()),
            self._output_directory_input.text().strip() or resolve_output_root(),
            checkpoint_path=checkpoint_path,
            stream_port=self._read_stream_port(),
        )
        self._append_log(f"[{self._format_time()}] Starting: {display_command}")

    def _read_stream_port(self) -> int | None:
        if self._streaming_server is None:
            return None
        return self._streaming_server.read_port()

    def _stop_simulation(self) -> None:
        if not self._runner.is_running():
            return
        self._append_log(f"[{self._format_time()}] Stopping…")
        self._runner.stop()

    def _handle_process_finished(self, code: int, status: str) -> None:
        self._clear_run_case_directory()
        self._run_button.setEnabled(True)
        self._resume_button.setEnabled(True)
        self._append_log(f"[{self._format_time()}] Finished with code {code} ({status})")

    def _handle_process_failed(self, message: str) -> None:
        self._clear_run_case_directory()
        self._run_button.setEnabled(True)
        self._resume_button.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "Failed to start", message)

    # ------------------------- File operations -------------------------
    def _open_file(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open result file",
            self._output_directory_input.text().strip() or os.getcwd(),
            "VTK rectilinear (*.vtr);;CSV totals (*_Total.csv);;All files (*)",
        )
        if not file_path:
            return
        self._load_file_path(file_path)

    def _refresh_watcher(self) -> None:
        self._watcher.set_directory(self._output_directory_input.text().strip())
        self._watcher.set_interval(self._session_settings.autoload_interval_ms)
        self._watcher.set_enabled(self._session_settings.autoload_enabled)
        if self._session_settings.autoload_enabled:
            self._watcher.poll()

    def _load_file_path(self, file_path: str) -> None:
        try:
            if file_path.lower().endswith(".vtr"):
                self._viewer.load_vtr(file_path)
                self._append_log(f"[{self._format_time()}] Loaded VTR: {file_path}{self._viewer.describe_time_level()}")
            elif file_path.lower().endswith("_total.csv"):
                self._viewer.load_csv(file_path)
                self._append_log(f"[{self._format_time()}] Loaded CSV: {file_path}{self._viewer.describe_time_level()}")
            else:
                QtWidgets.QMessageBox.warning(self, "Unsupported file", os.path.basename(file_path))
                return
            self._watcher.note_loaded(file_path)
            self._slice.refresh()
        except Exception:
            self._watcher.note_loaded(file_path)
            self._append_log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "Failed to load", os.path.basename(file_path))

    # ------------------------- Network operations -------------------------
    def _handle_streaming_dataset(self, dataset: pv.DataSet) -> None:
        # A live stream replaces the newest file on disk, so the watcher stops competing with it.
        self._watcher.set_enabled(False)
        try:
            self._viewer.load_mesh(dataset)
            self._slice.refresh()
            self._append_log(f"[{self._format_time()}] Streamed{self._viewer.describe_time_level()}")
        except Exception:
            self._append_log(traceback.format_exc())

    # ------------------------- Helpers -------------------------
    def _open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self, self._session_settings)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._session_settings = dialog.read_settings()
            self._refresh_watcher()

    def _open_simulation_params_dialog(self) -> None:
        dialog = SimulationParametersDialog(self, self._gui_case)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._gui_case = dialog.read_case()
            self._preset_input.setCurrentIndex(0)
            self._refresh_case_status()

    def _load_preset(self, index: int) -> None:
        preset_path = self._preset_input.itemData(index)
        if preset_path is None:
            return
        self._gui_case = read_case(preset_path)
        self._refresh_case_status()
        self._append_log(f"[{self._format_time()}] Loaded preset: {self._preset_input.itemText(index)}")

    def _refresh_case_status(self) -> None:
        shape = "x".join(str(count) for count in self._gui_case.grid.shape)
        self._params_status.setText(f"{shape}, {self._gui_case.outputs.format.value}")

    def _clear_run_case_directory(self) -> None:
        if self._run_case_directory is None:
            return
        self._run_case_directory.cleanup()
        self._run_case_directory = None

    def _browse_output_directory(self) -> None:
        directory_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose output directory",
            self._output_directory_input.text().strip() or os.getcwd(),
        )
        if directory_path:
            self._output_directory_input.setText(directory_path)
            self._refresh_watcher()

    def _show_help(self) -> None:
        """Show the help dialog."""
        help_dialog = HelpDialog(self)
        help_dialog.exec()

    def _append_log(self, text: str) -> None:
        self._log_output.appendPlainText(text)
        self._log_output.verticalScrollBar().setValue(self._log_output.verticalScrollBar().maximum())

    @staticmethod
    def _format_time() -> str:
        return datetime.now().strftime("%H:%M:%S")
