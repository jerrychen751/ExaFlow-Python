from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pytest
from PySide6 import QtWidgets

from exaflow.config import Case, CheckpointFormat
from exaflow.config.case_xml import read_case
from exaflow.gui.sim_parameters_dialog import SimulationParametersDialog, build_default_case

pytestmark = pytest.mark.gui

PRESET_PATHS = sorted(Path(str(files("exaflow.gui").joinpath("presets"))).glob("*.xml"))


@pytest.mark.parametrize("case", [build_default_case(), *(read_case(str(path)) for path in PRESET_PATHS)], ids=["default", *(path.stem for path in PRESET_PATHS)])
def test_the_dialog_returns_the_case_it_was_opened_on(qt_application: QtWidgets.QApplication, case: Case) -> None:
    assert SimulationParametersDialog(None, case).read_case() == case


def test_a_zero_count_leaves_the_axis_out(qt_application: QtWidgets.QApplication) -> None:
    dialog = SimulationParametersDialog(None, build_default_case())
    dialog._int_fields["domain_nz"].setValue(0)
    dialog._boundary_fields["left"].inflow["v"].setValue(0.5)

    case = dialog.read_case()

    assert case.grid.shape == (100, 100)
    assert case.grid.extent == (6.28, 3.14)
    assert case.boundaries.left.velocity == (2.0, 0.5)
    assert len(case.initial.velocity) == 2


def test_a_zero_count_before_a_set_one_is_refused(qt_application: QtWidgets.QApplication) -> None:
    dialog = SimulationParametersDialog(None, build_default_case())
    dialog._int_fields["domain_ny"].setValue(0)

    with pytest.raises(ValueError, match="nz must be 0 as well, got nz = 50"):
        dialog.read_case()


def test_the_checkpoint_format_can_be_changed(qt_application: QtWidgets.QApplication) -> None:
    dialog = SimulationParametersDialog(None, build_default_case())
    dialog._combo_fields["checkpoint_format"].setCurrentText("CSV")

    assert dialog.read_case().outputs.checkpoint_format is CheckpointFormat.CSV
