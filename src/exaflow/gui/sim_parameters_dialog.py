from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import xml.etree.ElementTree as ET

from PySide6 import QtWidgets

from ..config import (
    Boundaries,
    BoundaryCondition,
    Case,
    CheckpointFormat,
    Face,
    FaceCondition,
    Fluid,
    Grid,
    OutputControl,
    SolverOptions,
    TimeControl,
    parse_boundary_condition,
    parse_checkpoint_format,
)
from ..config.case_xml import parse_initial_conditions, write_initial_conditions


DEFAULT_INITIAL_CONDITIONS_XML = """<InitialConditions>
  <ReadFromVtrFile>False</ReadFromVtrFile>
  <ReadFromCsvFile>False</ReadFromCsvFile>
  <SpecifyValues>True</SpecifyValues>
  <SpecifiedValues>
    <p>
      <UseUniform>True</UseUniform>
      <UniformParameters>
        <ConstantValue>0</ConstantValue>
      </UniformParameters>
      <UseStep>False</UseStep>
      <UseSinusoidal>False</UseSinusoidal>
      <UsePolynomial>False</UsePolynomial>
      <UseGaussian>False</UseGaussian>
    </p>
    <u>
      <UseUniform>True</UseUniform>
      <UniformParameters>
        <ConstantValue>0</ConstantValue>
      </UniformParameters>
      <UseStep>False</UseStep>
      <UseSinusoidal>False</UseSinusoidal>
      <UsePolynomial>False</UsePolynomial>
      <UseGaussian>False</UseGaussian>
    </u>
    <v>
      <UseUniform>True</UseUniform>
      <UniformParameters>
        <ConstantValue>0</ConstantValue>
      </UniformParameters>
      <UseStep>False</UseStep>
      <UseSinusoidal>False</UseSinusoidal>
      <UsePolynomial>False</UsePolynomial>
      <UseGaussian>False</UseGaussian>
    </v>
    <w>
      <UseUniform>True</UseUniform>
      <UniformParameters>
        <ConstantValue>0</ConstantValue>
      </UniformParameters>
      <UseStep>False</UseStep>
      <UseSinusoidal>False</UseSinusoidal>
      <UsePolynomial>False</UsePolynomial>
      <UseGaussian>False</UseGaussian>
    </w>
  </SpecifiedValues>
</InitialConditions>
"""


def build_default_case() -> Case:
    """
    The case a dialog opens on when the caller gives it none. Every field the frozen types already default is left to them, so this states only what a new user has to be shown a value for.
    """

    return Case(
        fluid=Fluid(rho=1.225, nu=1.81e-5),
        grid=Grid(shape=(100, 100, 50), extent=(6.28, 3.14, 2.0), num_ghost_layers=1),
        time=TimeControl(num_steps=1000, cfl=0.5, integration_order=1),
        boundaries=Boundaries(
            left=FaceCondition(BoundaryCondition.INFLOW, (2.0, 1.0, 1.0)),
            right=FaceCondition(BoundaryCondition.OUTFLOW, pressure=0.0),
        ),
        initial=parse_initial_conditions(ET.fromstring(DEFAULT_INITIAL_CONDITIONS_XML), 3),
        outputs=OutputControl(checkpoint_format=CheckpointFormat.VTK, stream_frequency=100),
    )


@dataclass
class BoundaryWidgets:
    wall: QtWidgets.QLineEdit
    inflow: Dict[str, QtWidgets.QDoubleSpinBox]
    outflow: Dict[str, QtWidgets.QDoubleSpinBox]


class SimulationParametersDialog(QtWidgets.QDialog):
    """
    Dialog that edits one Case. It opens on the case it is given, and `read_case()` returns what the form now describes. The form cannot be accepted into a Case the solver would reject, because accepting builds that Case and reports whatever it raises.

    The dialog describes a 3D case only. It sets no rank count: the arrangement follows the communicator the run is launched with, which the main window sets.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None, initial_case: Case | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Simulation Parameters")
        self.setModal(True)
        self.resize(640, 720)

        self._double_fields: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._int_fields: Dict[str, QtWidgets.QSpinBox] = {}
        self._bool_fields: Dict[str, QtWidgets.QCheckBox] = {}
        self._combo_fields: Dict[str, QtWidgets.QComboBox] = {}
        self._boundary_fields: Dict[str, BoundaryWidgets] = {}

        self._initial_conditions_editor = QtWidgets.QPlainTextEdit()

        self._build_ui()
        self._populate_from_case(initial_case or build_default_case())

    # ------------------------ UI construction ------------------------ #
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(tabs, 1)

        tabs.addTab(self._build_fluid_tab(), "Fluid & Grid")
        tabs.addTab(self._build_solver_tab(), "Solver")
        tabs.addTab(self._build_boundaries_tab(), "Boundaries")
        tabs.addTab(self._build_output_tab(), "Output & Misc")

        initial_conditions_tab = QtWidgets.QWidget()
        initial_conditions_layout = QtWidgets.QVBoxLayout(initial_conditions_tab)
        initial_conditions_layout.addWidget(QtWidgets.QLabel("Provide a full <InitialConditions> XML block:"))
        self._initial_conditions_editor.setPlaceholderText("<InitialConditions>…</InitialConditions>")
        self._initial_conditions_editor.setTabChangesFocus(True)
        initial_conditions_layout.addWidget(self._initial_conditions_editor, 1)
        tabs.addTab(initial_conditions_tab, "Initial Conditions")

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._handle_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _build_fluid_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self._double_fields["rho"] = self._create_double_input(minimum=1e-9, maximum=1e6, decimals=6)
        form.addRow("Density (rho)", self._double_fields["rho"])

        self._double_fields["nu"] = self._create_double_input(minimum=0.0, maximum=1e6, decimals=8, single_step=1e-6)
        form.addRow("Kinematic viscosity (nu)", self._double_fields["nu"])

        self._int_fields["domain_nx"] = self._create_int_input(2, 10000)
        self._int_fields["domain_ny"] = self._create_int_input(0, 10000)
        self._int_fields["domain_nz"] = self._create_int_input(0, 10000)
        for axis in ("ny", "nz"):
            self._int_fields[f"domain_{axis}"].setToolTip("0 leaves this axis out, so ny = nz = 0 is a 1D case and nz = 0 is a 2D case.")
        form.addRow("Domain (nx, ny, nz)", self._merge_triplet(self._int_fields["domain_nx"], self._int_fields["domain_ny"], self._int_fields["domain_nz"]))

        self._double_fields["size_length"] = self._create_double_input(1e-6, 1e6, 3, 0.1)
        self._double_fields["size_width"] = self._create_double_input(1e-6, 1e6, 3, 0.1)
        self._double_fields["size_height"] = self._create_double_input(1e-6, 1e6, 3, 0.1)
        form.addRow("Size (L, W, H)", self._merge_triplet(self._double_fields["size_length"], self._double_fields["size_width"], self._double_fields["size_height"]))

        self._int_fields["nt"] = self._create_int_input(1, 10_000_000)
        form.addRow("Time steps (nt)", self._int_fields["nt"])

        self._int_fields["num_ghost_layers"] = self._create_int_input(1, 10)
        form.addRow("Ghost layers", self._int_fields["num_ghost_layers"])

        self._double_fields["cfl"] = self._create_double_input(1e-6, 10.0, 4, 0.05)
        form.addRow("CFL", self._double_fields["cfl"])

        self._double_fields["end_time"] = self._create_double_input(0.0, 1e9, 6, 0.1)
        form.addRow("End time (s, 0 for none)", self._double_fields["end_time"])

        self._bool_fields["adaptive_time_step"] = QtWidgets.QCheckBox("Adaptive time step")
        form.addRow(self._bool_fields["adaptive_time_step"])

        return widget

    def _build_solver_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self._bool_fields["include_convection"] = QtWidgets.QCheckBox("Include convection")
        form.addRow(self._bool_fields["include_convection"])
        self._combo_fields["convection_scheme"] = self._create_scheme_combo(["Upwind"])
        form.addRow("Convection scheme", self._combo_fields["convection_scheme"])

        self._bool_fields["include_diffusion"] = QtWidgets.QCheckBox("Include diffusion")
        form.addRow(self._bool_fields["include_diffusion"])
        self._combo_fields["viscous_scheme"] = self._create_scheme_combo(["CentralDifference"])
        form.addRow("Viscous scheme", self._combo_fields["viscous_scheme"])

        self._bool_fields["include_pressure"] = QtWidgets.QCheckBox("Include pressure")
        form.addRow(self._bool_fields["include_pressure"])

        self._int_fields["time_integration_order"] = self._create_int_input(1, 3)
        form.addRow("Time integration order", self._int_fields["time_integration_order"])

        return widget

    def _build_boundaries_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid, 1)

        boundary_names = [
            ("left", "Left"),
            ("right", "Right"),
            ("top", "Top"),
            ("bottom", "Bottom"),
            ("front", "Front"),
            ("back", "Back"),
        ]

        for index, (key, title) in enumerate(boundary_names):
            group = QtWidgets.QGroupBox(f"{title} Boundary")
            group_layout = QtWidgets.QFormLayout(group)

            wall_field = QtWidgets.QLineEdit()
            group_layout.addRow("Wall type", wall_field)

            inflow_u = self._create_double_input(-1e3, 1e3, 4, 0.1)
            inflow_v = self._create_double_input(-1e3, 1e3, 4, 0.1)
            inflow_w = self._create_double_input(-1e3, 1e3, 4, 0.1)
            inflow_layout = self._merge_triplet(inflow_u, inflow_v, inflow_w, labels=("u", "v", "w"))
            group_layout.addRow("Inflow (u,v,w)", inflow_layout)

            outflow_p = self._create_double_input(-1e6, 1e6, 4, 0.1)
            outflow_layout = QtWidgets.QHBoxLayout()
            outflow_layout.addWidget(QtWidgets.QLabel("p"))
            outflow_layout.addWidget(outflow_p)
            group_layout.addRow("Outflow", outflow_layout)

            row = index // 2
            col = index % 2
            grid.addWidget(group, row, col)

            self._boundary_fields[key] = BoundaryWidgets(
                wall=wall_field,
                inflow={"u": inflow_u, "v": inflow_v, "w": inflow_w},
                outflow={"p": outflow_p},
            )

        layout.addStretch(1)
        return widget

    def _build_output_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)

        self._combo_fields["checkpoint_format"] = self._create_scheme_combo([fmt.value for fmt in CheckpointFormat])
        form.addRow("Checkpoint format", self._combo_fields["checkpoint_format"])

        self._int_fields["stream_frequency"] = self._create_int_input(-1, 1_000_000)
        form.addRow("Stream frequency", self._int_fields["stream_frequency"])

        self._int_fields["checkpoint_frequency"] = self._create_int_input(-1, 1_000_000)
        form.addRow("Checkpoint frequency", self._int_fields["checkpoint_frequency"])

        return widget

    # ------------------------ Helpers ------------------------ #
    def _create_double_input(
        self,
        minimum: float = -1e9,
        maximum: float = 1e9,
        decimals: int = 6,
        single_step: float = 0.01,
    ) -> QtWidgets.QDoubleSpinBox:
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(single_step)
        widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.PlusMinus)
        return widget

    def _create_int_input(self, minimum: int, maximum: int) -> QtWidgets.QSpinBox:
        widget = QtWidgets.QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.PlusMinus)
        widget.setAccelerated(True)
        return widget

    @staticmethod
    def _merge_triplet(
        first: QtWidgets.QWidget,
        second: QtWidgets.QWidget,
        third: QtWidgets.QWidget,
        labels: tuple[str, str, str] | None = None,
    ) -> QtWidgets.QWidget:
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        widgets: List[QtWidgets.QWidget] = [first, second, third]
        for index, widget in enumerate(widgets):
            if labels:
                layout.addWidget(QtWidgets.QLabel(labels[index]))
            layout.addWidget(widget)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        return container

    @staticmethod
    def _create_scheme_combo(options: List[str]) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItems(options)
        return combo

    def _populate_from_case(self, case: Case) -> None:
        self._double_fields["rho"].setValue(case.fluid.rho)
        self._double_fields["nu"].setValue(case.fluid.nu)

        for index, axis in enumerate(("nx", "ny", "nz")):
            self._int_fields[f"domain_{axis}"].setValue(case.grid.shape[index] if index < case.dimension else 0)
        for index, name in enumerate(("size_length", "size_width", "size_height")[: case.dimension]):
            self._double_fields[name].setValue(case.grid.extent[index])

        self._int_fields["nt"].setValue(case.time.num_steps)
        self._int_fields["num_ghost_layers"].setValue(case.grid.num_ghost_layers)
        self._double_fields["cfl"].setValue(case.time.cfl)
        self._double_fields["end_time"].setValue(0.0 if case.time.end_time is None else case.time.end_time)
        self._bool_fields["adaptive_time_step"].setChecked(case.time.adaptive_time_step)

        self._bool_fields["include_convection"].setChecked(case.solver.include_convection)
        self._combo_fields["convection_scheme"].setCurrentText(case.solver.convection_scheme)
        self._bool_fields["include_diffusion"].setChecked(case.solver.include_diffusion)
        self._combo_fields["viscous_scheme"].setCurrentText(case.solver.viscous_scheme)
        self._bool_fields["include_pressure"].setChecked(case.solver.include_pressure)
        self._int_fields["time_integration_order"].setValue(case.time.integration_order)

        self._combo_fields["checkpoint_format"].setCurrentText(case.outputs.checkpoint_format.value)
        self._int_fields["stream_frequency"].setValue(case.outputs.stream_frequency)
        self._int_fields["checkpoint_frequency"].setValue(case.outputs.checkpoint_frequency)

        for face in Face:
            condition = case.boundaries.find_face(face)
            widgets = self._boundary_fields[face.name.lower()]
            widgets.wall.setText(condition.kind.value)
            for index, name in enumerate(("u", "v", "w")):
                widgets.inflow[name].setValue(condition.velocity[index] if index < len(condition.velocity) else 0.0)
            widgets.outflow["p"].setValue(condition.pressure)

        self._initial_conditions_editor.setPlainText(write_initial_conditions(case.initial, case.dimension))

    # ------------------------ Data extraction ------------------------ #
    def read_case(self) -> Case:
        """
        The case the form now describes. Raises ValueError or NotImplementedError when the form describes a case the solver rejects, so the caller reports the fault here instead of writing an XML file that fails at run time.
        """

        counts = [self._int_fields[f"domain_{axis}"].value() for axis in ("nx", "ny", "nz")]
        if counts[1] == 0 and counts[2] > 0:
            raise ValueError(f"ny = 0 leaves the y axis out, so nz must be 0 as well, got nz = {counts[2]}. A 2D case uses nx and ny.")
        dimension = sum(1 for count in counts if count > 0)

        faces = {}
        for face in Face:
            widgets = self._boundary_fields[face.name.lower()]
            kind = parse_boundary_condition(widgets.wall.text().strip() or BoundaryCondition.NO_SLIP.value)
            velocity: tuple[float, ...] = ()
            if kind == BoundaryCondition.INFLOW:
                velocity = tuple(widgets.inflow[name].value() for name in ("u", "v", "w")[:dimension])
            pressure = widgets.outflow["p"].value() if kind == BoundaryCondition.OUTFLOW else 0.0
            faces[face.name.lower()] = FaceCondition(kind=kind, velocity=velocity, pressure=pressure)

        initial_text = self._initial_conditions_editor.toPlainText().strip() or DEFAULT_INITIAL_CONDITIONS_XML
        try:
            initial_element = ET.fromstring(initial_text)
        except ET.ParseError as error:
            raise ValueError(f"The initial conditions block is not valid XML: {error}") from error

        return Case(
            fluid=Fluid(
                rho=self._double_fields["rho"].value(),
                nu=self._double_fields["nu"].value(),
            ),
            grid=Grid(
                shape=tuple(counts[:dimension]),
                extent=tuple(self._double_fields[name].value() for name in ("size_length", "size_width", "size_height")[:dimension]),
                num_ghost_layers=self._int_fields["num_ghost_layers"].value(),
            ),
            time=TimeControl(
                num_steps=self._int_fields["nt"].value(),
                cfl=self._double_fields["cfl"].value(),
                integration_order=self._int_fields["time_integration_order"].value(),
                end_time=self._double_fields["end_time"].value() or None,
                adaptive_time_step=self._bool_fields["adaptive_time_step"].isChecked(),
            ),
            boundaries=Boundaries(**faces),
            initial=parse_initial_conditions(initial_element, dimension),
            solver=SolverOptions(
                include_convection=self._bool_fields["include_convection"].isChecked(),
                include_diffusion=self._bool_fields["include_diffusion"].isChecked(),
                include_pressure=self._bool_fields["include_pressure"].isChecked(),
                convection_scheme=self._combo_fields["convection_scheme"].currentText().strip(),
                viscous_scheme=self._combo_fields["viscous_scheme"].currentText().strip(),
            ),
            outputs=OutputControl(
                checkpoint_format=parse_checkpoint_format(self._combo_fields["checkpoint_format"].currentText().strip()),
                stream_frequency=self._int_fields["stream_frequency"].value(),
                checkpoint_frequency=self._int_fields["checkpoint_frequency"].value(),
            ),
        )

    # ------------------------ Validation ------------------------ #
    def _handle_accept(self) -> None:
        """
        Accept only a form that builds a Case. Every rule lives in the frozen types, so the dialog states none of them a second time and cannot disagree with the solver about what is valid.
        """

        try:
            self.read_case()
        except (ValueError, NotImplementedError) as error:
            QtWidgets.QMessageBox.warning(self, "Invalid parameters", str(error))
            return
        self.accept()
