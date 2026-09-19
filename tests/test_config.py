from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Callable

import pytest

from exaflow.config import parse_boundary_condition
from exaflow.config import (
    Boundaries,
    BoundaryCondition,
    Case,
    CheckpointFormat,
    Face,
    FaceCondition,
    Fluid,
    Grid,
    InitialConditions,
    OutputControl,
    SolverOptions,
    StepValue,
    TimeControl,
    UniformValue,
    collect_faces,
)


def test_grid_spacing_places_a_point_on_each_end() -> None:
    grid = Grid((5, 3), (1.0, 2.0), 1)
    assert grid.spacing == (0.25, 1.0)
    assert grid.dimension == 2


def test_case_is_frozen(build_case: Callable[..., Case]) -> None:
    case = build_case()
    with pytest.raises(FrozenInstanceError):
        case.grid = Grid((4, 4, 4), (1.0, 1.0, 1.0), 1)  # type: ignore[misc]


def test_grid_rejects_an_axis_with_one_point() -> None:
    with pytest.raises(ValueError, match="at least 2 grid points"):
        Grid((1, 4), (1.0, 1.0), 1)


@pytest.mark.parametrize(
    "shape,extent,message",
    [
        ((2, 2, 2, 2), (1.0, 1.0, 1.0, 1.0), "1, 2 or 3 axes"),
        ((4, 4), (1.0,), "must match shape in length"),
        ((4, 4), (1.0, 0.0), "finite and > 0"),
        ((4, 4), (1.0, float("inf")), "finite and > 0"),
    ],
)
def test_grid_rejects_a_malformed_shape_or_extent(
    shape: tuple[int, ...],
    extent: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Grid(shape, extent, 1)


def test_grid_needs_at_least_one_ghost_layer() -> None:
    with pytest.raises(ValueError, match="num_ghost_layers must be >= 1"):
        Grid((4, 4), (1.0, 1.0), 0)


@pytest.mark.parametrize("rho,nu,message", [(0.0, 1.0, "rho"), (-1.0, 1.0, "rho"), (1.0, -1e-9, "nu")])
def test_fluid_rejects_an_impossible_property(rho: float, nu: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Fluid(rho, nu)


def test_an_inviscid_fluid_is_allowed() -> None:
    assert Fluid(1.0, 0.0).nu == 0.0


@pytest.mark.parametrize(
    "num_steps,cfl,order,message",
    [
        (0, 0.25, 1, "num_steps"),
        (-1, 0.25, 1, "num_steps"),
        (1, 0.0, 1, "cfl"),
        (1, float("nan"), 1, "cfl"),
        (1, 0.25, 4, "integration_order"),
        (1, 0.25, 0, "integration_order"),
    ],
)
def test_time_control_rejects_a_setting_the_march_cannot_use(
    num_steps: int,
    cfl: float,
    order: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TimeControl(num_steps, cfl, order)


@pytest.mark.parametrize("end_time", [0.0, -1.0, float("nan"), float("inf")])
def test_time_control_rejects_an_end_time_the_march_cannot_reach(end_time: float) -> None:
    with pytest.raises(ValueError, match="end_time must be finite and > 0"):
        TimeControl(10, 0.25, 1, end_time=end_time)


def test_a_run_with_no_end_time_marches_on_the_step_budget_alone() -> None:
    control = TimeControl(10, 0.25)
    assert control.end_time is None
    assert control.adaptive_time_step is False


def test_output_control_rejects_a_zero_checkpoint_interval() -> None:
    with pytest.raises(ValueError, match="checkpoint_frequency must be -1 or >= 1"):
        OutputControl(checkpoint_frequency=0)


def test_output_control_rejects_a_zero_interval() -> None:
    with pytest.raises(ValueError, match="must be -1 or >= 1"):
        OutputControl(stream_frequency=0)


def test_output_control_selects_the_checkpoint_format() -> None:
    assert OutputControl(checkpoint_format=CheckpointFormat.VTK).checkpoint_format is CheckpointFormat.VTK


def test_output_control_selects_the_right_steps() -> None:
    outputs = OutputControl(stream_frequency=10, checkpoint_frequency=25)
    assert outputs.is_due(outputs.stream_frequency, 0)
    assert outputs.is_due(outputs.stream_frequency, 20)
    assert not outputs.is_due(outputs.stream_frequency, 15)
    assert outputs.is_due(outputs.checkpoint_frequency, 50)


def test_an_interval_of_minus_one_is_due_at_no_step() -> None:
    """
    -1 selects no intermediate stream and no checkpoint.
    """

    outputs = OutputControl()
    assert not any(outputs.is_due(outputs.stream_frequency, step) for step in range(50))
    assert not any(outputs.is_due(outputs.checkpoint_frequency, step) for step in range(50))


def test_periodic_faces_must_be_paired() -> None:
    with pytest.raises(ValueError, match="Periodic faces must be paired"):
        Boundaries(left=FaceCondition(BoundaryCondition.PERIODIC))


def test_a_periodic_pair_reports_its_axis() -> None:
    periodic = FaceCondition(BoundaryCondition.PERIODIC)
    boundaries = Boundaries(front=periodic, back=periodic)
    assert boundaries.is_periodic(2)
    assert not boundaries.is_periodic(0)
    assert not boundaries.is_periodic(1)


def test_an_inflow_face_needs_a_velocity() -> None:
    with pytest.raises(ValueError, match="needs a velocity tuple"):
        FaceCondition(BoundaryCondition.INFLOW)


def test_a_time_dependent_face_is_refused_rather_than_ignored() -> None:
    with pytest.raises(NotImplementedError, match="Time-dependent"):
        FaceCondition(BoundaryCondition.TIME_DEPENDENT)


def test_every_face_has_an_opposite_on_its_own_axis() -> None:
    for face in Face:
        assert face.opposite.axis == face.axis
        assert face.opposite.is_low != face.is_low
        assert face.opposite.opposite is face


@pytest.mark.parametrize(
    "dimension,expected",
    [
        (1, (Face.LEFT, Face.RIGHT)),
        (2, (Face.LEFT, Face.RIGHT, Face.TOP, Face.BOTTOM)),
        (3, tuple(Face)),
    ],
)
def test_collect_faces_stops_at_the_case_dimension(dimension: int, expected: tuple[Face, ...]) -> None:
    assert collect_faces(dimension) == expected


def test_inflow_velocity_must_match_the_dimension(build_case: Callable[..., Case]) -> None:
    with pytest.raises(ValueError, match="velocity components"):
        build_case(
            (4, 4),
            boundaries=Boundaries(left=FaceCondition(BoundaryCondition.INFLOW, (1.0, 1.0, 1.0))),
        )


def test_initial_velocity_must_have_one_entry_per_axis(build_case: Callable[..., Case]) -> None:
    with pytest.raises(ValueError, match="one entry per axis"):
        build_case((4, 4), initial=InitialConditions(velocity=((UniformValue(1.0),),)))


def test_an_inflow_face_on_an_axis_the_case_does_not_have_is_ignored(build_case: Callable[..., Case]) -> None:
    """
    A 2D case still carries FRONT and BACK on the Boundaries record, because Boundaries always holds six faces. Only the faces collect_faces returns are checked, so a leftover 3D inflow on FRONT cannot fail a 2D case.
    """

    case = build_case(
        (4, 4),
        boundaries=Boundaries(front=FaceCondition(BoundaryCondition.INFLOW, (1.0, 1.0, 1.0))),
    )
    assert case.dimension == 2


@pytest.mark.parametrize(
    "overrides,message",
    [
        (dict(include_pressure=True), "Pressure projection"),
        (dict(convection_scheme="QUICK"), "convection_scheme"),
        (dict(viscous_scheme="Compact"), "viscous_scheme"),
    ],
)
def test_solver_options_refuse_a_scheme_that_is_not_implemented(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(NotImplementedError, match=message):
        SolverOptions(**overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "start,end,message",
    [
        ((0.0, 0.0), (1.0,), "match in length"),
        ((0.0,), (1.5,), r"within \[0, 1\]"),
        ((-0.1,), (1.0,), r"within \[0, 1\]"),
        ((0.8,), (0.2,), "end must be >= start"),
        ((float("nan"),), (1.0,), "must be finite"),
    ],
)
def test_step_value_rejects_a_box_it_cannot_place(
    start: tuple[float, ...],
    end: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        StepValue(1.0, start, end)


def test_an_unknown_boundary_condition_names_the_allowed_values() -> None:
    with pytest.raises(ValueError, match="Unknown boundary condition 'Sponge'"):
        parse_boundary_condition("Sponge")


@pytest.mark.parametrize("condition", list(BoundaryCondition))
def test_every_boundary_condition_parses_back_from_its_own_value(condition: BoundaryCondition) -> None:
    assert parse_boundary_condition(condition.value) is condition
