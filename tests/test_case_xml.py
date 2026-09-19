from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

from exaflow.config import (
    Boundaries,
    BoundaryCondition,
    Case,
    CheckpointFormat,
    FaceCondition,
    Fluid,
    Grid,
    InitialConditions,
    OutputControl,
    StepValue,
    TimeControl,
    UniformValue,
)
from exaflow.config.case_xml import parse_case, parse_initial_conditions, read_case, write_case


def build_full_case(dimension: int) -> Case:
    """
    A case that exercises every part the XML format carries: a non-default ghost depth, an inflow face, an outflow face, a step box on each velocity component and a uniform pressure.
    """

    shape = (6, 5, 4)[:dimension]
    extent = (1.0, 2.0, 3.0)[:dimension]
    return Case(
        fluid=Fluid(1.2, 0.01),
        grid=Grid(shape, extent, 2),
        time=TimeControl(9, 0.3, 2),
        boundaries=Boundaries(
            left=FaceCondition(BoundaryCondition.INFLOW, tuple(float(axis) for axis in range(dimension))),
            right=FaceCondition(BoundaryCondition.OUTFLOW, pressure=1.25),
        ),
        initial=InitialConditions(
            pressure=(UniformValue(0.5),),
            velocity=tuple(
                (StepValue(2.0, (0.25,) * dimension, (0.75,) * dimension),) for _ in range(dimension)
            ),
        ),
        outputs=OutputControl(checkpoint_format=CheckpointFormat.VTK, stream_frequency=2),
    )


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_a_case_of_every_dimension_round_trips(dimension: int) -> None:
    """
    The writer pads the point count and the span of every absent axis with zero, and the reader takes the dimension from the count of positive nx, ny and nz. A 1D or 2D case therefore has to come back at the dimension it went in.
    """

    case = build_full_case(dimension)
    assert parse_case(ElementTree.fromstring(write_case(case))) == case


def test_the_shipped_template_round_trips(template_case_path: Path) -> None:
    case = read_case(str(template_case_path))
    assert parse_case(ElementTree.fromstring(write_case(case))) == case


def test_reading_the_shipped_template_gives_the_documented_case(template_case_path: Path) -> None:
    case = read_case(str(template_case_path))
    assert case.grid.shape == (100, 100, 50)
    assert case.time.num_steps == 1000
    assert case.fluid.nu == pytest.approx(1.81e-5)


def test_several_uniform_contributions_are_written_as_their_sum() -> None:
    """
    The format holds one uniform value per field, so the writer adds them. The field starts at the same value and reads back as one contribution.
    """

    case = Case(
        fluid=Fluid(1.0, 0.1),
        grid=Grid((4,), (1.0,), 1),
        time=TimeControl(1, 0.25),
        initial=InitialConditions(pressure=(UniformValue(1.0), UniformValue(2.5))),
    )
    assert parse_case(ElementTree.fromstring(write_case(case))).initial.pressure == (UniformValue(3.5),)


def test_a_second_step_contribution_is_refused_rather_than_dropped() -> None:
    case = Case(
        fluid=Fluid(1.0, 0.1),
        grid=Grid((4,), (1.0,), 1),
        time=TimeControl(1, 0.25),
        initial=InitialConditions(
            pressure=(StepValue(1.0, (0.0,), (0.5,)), StepValue(1.0, (0.5,), (1.0,))),
        ),
    )
    with pytest.raises(ValueError, match="2 step contributions"):
        write_case(case)


def test_a_root_element_that_is_not_a_simulation_is_refused() -> None:
    with pytest.raises(ValueError, match="Expected a <Simulation> root element"):
        parse_case(ElementTree.fromstring("<Case/>"))


def read_template_root(path: Path) -> ElementTree.Element:
    return ElementTree.parse(path).getroot()


@pytest.mark.parametrize(
    "parent_tag,child_tag",
    [
        ("Simulation", "FluidProperties"),
        ("Simulation", "OutputProperties"),
        ("OutputProperties", "CheckpointFormat"),
        ("GridProperties", "Domain"),
        ("GridProperties", "EndTime"),
        ("GridProperties", "AdaptiveTimeStep"),
        ("OutputProperties", "StreamFrequency"),
        ("OutputProperties", "CheckpointFrequency"),
        ("BoundaryConditions", "LeftWall"),
    ],
)
def test_a_missing_element_names_the_parent_and_the_tag(
    template_case_path: Path,
    parent_tag: str,
    child_tag: str,
) -> None:
    """
    The reader fills in no default. An input file that omits an element is a file whose author expected something the run would not do.
    """

    root = read_template_root(template_case_path)
    parent = root if parent_tag == "Simulation" else next(root.iter(parent_tag))
    parent.remove(next(parent.iter(child_tag)))
    with pytest.raises(ValueError, match=f"<{parent_tag}> is missing the <{child_tag}> element"):
        parse_case(root)


@pytest.mark.parametrize(
    "tag,text,message",
    [
        ("nx", "one hundred", "<nx> must be an integer"),
        ("Rho", "dense", "<Rho> must be a number"),
        ("IncludeConvectionEffects", "yes", "<IncludeConvectionEffects> must be True or False"),
        ("LeftWall", "Sponge", "Unknown boundary condition 'Sponge'"),
        ("CheckpointFormat", "HDF5", "Unknown checkpoint format 'HDF5'"),
    ],
)
def test_a_value_of_the_wrong_type_names_its_tag(
    template_case_path: Path,
    tag: str,
    text: str,
    message: str,
) -> None:
    root = read_template_root(template_case_path)
    next(root.iter(tag)).text = text
    with pytest.raises(ValueError, match=message):
        parse_case(root)


def test_a_domain_with_no_positive_point_count_is_refused(template_case_path: Path) -> None:
    root = read_template_root(template_case_path)
    for tag in ("nx", "ny", "nz"):
        next(root.iter(tag)).text = "0"
    with pytest.raises(ValueError, match="Domain must give a positive nx"):
        parse_case(root)


@pytest.mark.parametrize(
    "tag,message",
    [
        ("ReadFromVtrFile", "VTR file is not implemented"),
        ("ReadFromCsvFile", "CSV file is not implemented"),
    ],
)
def test_an_initial_condition_read_from_a_file_is_refused(
    template_case_path: Path,
    tag: str,
    message: str,
) -> None:
    node = next(read_template_root(template_case_path).iter("InitialConditions"))
    next(node.iter(tag)).text = "True"
    with pytest.raises(NotImplementedError, match=message):
        parse_initial_conditions(node, 3)


def test_initial_conditions_must_specify_values(template_case_path: Path) -> None:
    node = next(read_template_root(template_case_path).iter("InitialConditions"))
    next(node.iter("SpecifyValues")).text = "False"
    with pytest.raises(ValueError, match="must set SpecifyValues to True"):
        parse_initial_conditions(node, 3)


@pytest.mark.parametrize(
    "tag,message",
    [
        ("UseSinusoidal", "Sinusoidal initial conditions"),
        ("UsePolynomial", "Polynomial initial conditions"),
        ("UseGaussian", "Gaussian initial conditions"),
    ],
)
def test_an_unimplemented_initial_shape_is_refused(
    template_case_path: Path,
    tag: str,
    message: str,
) -> None:
    node = next(read_template_root(template_case_path).iter("InitialConditions"))
    next(next(node.iter("SpecifiedValues")).iter(tag)).text = "True"
    with pytest.raises(NotImplementedError, match=message):
        parse_initial_conditions(node, 3)


@pytest.mark.parametrize("text,expected", [("True", True), ("true", True), ("1", True), ("False", False), ("0", False)])
def test_a_boolean_reads_in_every_spelling_the_format_allows(
    template_case_path: Path,
    text: str,
    expected: bool,
) -> None:
    root = read_template_root(template_case_path)
    next(root.iter("IncludeViscousEffects")).text = text
    assert parse_case(root).solver.include_diffusion is expected
