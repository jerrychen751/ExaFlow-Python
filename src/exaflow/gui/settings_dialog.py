from __future__ import annotations

from dataclasses import dataclass
from PySide6 import QtWidgets


@dataclass
class SessionSettings:
    autoload_enabled: bool = True
    autoload_interval_ms: int = 2000


class SettingsDialog(QtWidgets.QDialog):
    """Transient settings dialog for session-only viewer preferences."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, settings: SessionSettings | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(400, 180)

        self._initial_settings = settings or SessionSettings()

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self._autoload_checkbox = QtWidgets.QCheckBox("Auto-load newest checkpoint")
        self._autoload_checkbox.setChecked(self._initial_settings.autoload_enabled)
        form.addRow("Auto-load", self._autoload_checkbox)

        self._autoload_interval_spin = QtWidgets.QSpinBox()
        self._autoload_interval_spin.setRange(250, 60_000)
        self._autoload_interval_spin.setSuffix(" ms")
        self._autoload_interval_spin.setSingleStep(250)
        self._autoload_interval_spin.setValue(self._initial_settings.autoload_interval_ms)
        form.addRow("Auto-load interval", self._autoload_interval_spin)

        self._autoload_checkbox.toggled.connect(self._autoload_interval_spin.setEnabled)
        self._autoload_interval_spin.setEnabled(self._autoload_checkbox.isChecked())

        self._note_label = QtWidgets.QLabel("Settings apply to this session only.")
        layout.addWidget(self._note_label)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def read_settings(self) -> SessionSettings:
        return SessionSettings(
            autoload_enabled=self._autoload_checkbox.isChecked(),
            autoload_interval_ms=int(self._autoload_interval_spin.value()),
        )
