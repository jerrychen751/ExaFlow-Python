from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from exaflow.config import (
    Boundaries,
    BoundaryCondition,
    Case,
    FaceCondition,
    Fluid,
    Grid,
    InitialConditions,
    OutputControl,
    OutputFormat,
    TimeControl,
    UniformValue,
)
from exaflow.session import SimulationSession


@pytest.fixture(scope="session")
def moving_case(build_case: Callable[..., Case]) -> Case:
    """
    A 3D case that marches five steps with an inflow on the left face and a uniform starting velocity, so every term of the right-hand side has something to act on.
    """

    return build_case(
        (12, 12, 12),
        time=TimeControl(5, 0.25, 1),
        boundaries=Boundaries(left=FaceCondition(BoundaryCondition.INFLOW, (2.0, 1.0, 1.0))),
        initial=InitialConditions(velocity=tuple((UniformValue(1.0),) for _ in range(3))),
    )


def test_a_serial_run_stays_finite_and_respects_the_inflow(moving_case: Case) -> None:
    session = SimulationSession(moving_case)
    state = session.run_until_complete()
    assert np.all(np.isfinite(state.velocity))
    assert np.allclose(state.velocity[0][0, 1:-1, 1:-1], 2.0)


def test_a_session_with_no_destination_writes_nothing(moving_case: Case) -> None:
    assert SimulationSession(moving_case).writers == ()


def test_explicit_writers_replace_the_standard_set(tmp_path: Path, moving_case: Case) -> None:
    from exaflow.io.writers import RankCsvWriter

    session = SimulationSession(moving_case, output_directory=str(tmp_path), writers=())
    assert session.writers == ()

    session = SimulationSession(moving_case, writers=(RankCsvWriter(str(tmp_path), session.subdomain),))
    assert len(session.writers) == 1


def test_a_stream_port_adds_a_stream_writer_at_the_whole_domain_interval(moving_case: Case) -> None:
    from exaflow.io.writers import StreamWriter

    case = replace(moving_case, outputs=OutputControl(total_frequency=3))
    session = SimulationSession(case, writers=(), stream_port=12345)

    assert [type(writer) for writer in session.writers] == [StreamWriter]
    assert session.writers[0].frequency == 3


def test_the_position_starts_at_zero_and_moves_with_every_step(moving_case: Case) -> None:
    session = SimulationSession(moving_case)
    assert (session.step_index, session.current_time) == (0, 0.0)

    session.advance_one_step()

    assert session.step_index == 1
    assert session.current_time == pytest.approx(session.dt)
    assert session.level.step_index == 1


def test_one_step_at_a_time_gives_the_same_answer_as_one_call(moving_case: Case) -> None:
    """
    A caller that stops between steps has to reach the state a whole run reaches, or a restart would answer a different question from the run it continues.
    """

    whole = SimulationSession(moving_case).run_until_complete()

    stepped = SimulationSession(moving_case)
    for _ in range(moving_case.time.num_steps):
        stepped.advance_one_step()

    assert np.array_equal(stepped.state.velocity, whole.velocity)
    assert np.array_equal(stepped.state.pressure, whole.pressure)
    assert stepped.current_time == pytest.approx(moving_case.time.num_steps * stepped.dt)


def test_a_run_that_has_reached_its_target_refuses_another_step(moving_case: Case) -> None:
    """
    The step size of a run at its end time is zero or less, so the run would march backwards rather than stand still.
    """

    session = SimulationSession(moving_case)
    session.run_until_complete()

    assert session.is_complete()
    with pytest.raises(RuntimeError, match="complete at step 5 of 5"):
        session.advance_one_step()


def test_an_end_time_stops_the_run_on_the_exact_second(build_case: Callable[..., Case]) -> None:
    """
    The last step is cut to the time that is left, and the run then holds the end time itself. A sum of steps would land one unit in the last place away from it and take one more step of nothing.
    """

    case = build_case(
        (6, 6),
        time=TimeControl(100, 0.25, 1),
        initial=InitialConditions(velocity=((UniformValue(1.0),), (UniformValue(0.5),))),
    )
    step_size = SimulationSession(case).dt
    timed = replace(case, time=TimeControl(100, 0.25, 1, end_time=2.5 * step_size))

    session = SimulationSession(timed)
    session.run_until_complete()

    assert session.current_time == timed.time.end_time
    assert session.step_index == 3


def test_the_step_budget_caps_a_run_that_would_not_reach_its_end_time(
    build_case: Callable[..., Case],
) -> None:
    case = build_case(
        (6, 6),
        time=TimeControl(3, 0.25, 1, end_time=1.0e6),
        initial=InitialConditions(velocity=((UniformValue(1.0),), (UniformValue(0.5),))),
    )

    session = SimulationSession(case)
    session.run_until_complete()

    assert session.step_index == 3
    assert session.current_time < 1.0e6


def test_an_adaptive_run_chooses_the_step_size_again_before_every_step(
    build_case: Callable[..., Case],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The choice costs one reduction across the ranks, so a fixed run pays for it once, when it builds the session, and an adaptive run pays once more for every step.
    """

    calls = []
    original = SimulationSession.choose_time_step

    def record(session: SimulationSession) -> float:
        calls.append(session)
        return original(session)

    monkeypatch.setattr(SimulationSession, "choose_time_step", record)
    case = build_case(
        (6, 6),
        time=TimeControl(3, 0.25, 1, adaptive_time_step=True),
        initial=InitialConditions(velocity=((UniformValue(1.0),), (UniformValue(0.5),))),
    )

    SimulationSession(case).run_until_complete()
    assert len(calls) == 4

    calls.clear()
    SimulationSession(replace(case, time=TimeControl(3, 0.25, 1))).run_until_complete()
    assert len(calls) == 1


def test_a_checkpoint_interval_with_nowhere_to_write_is_refused(build_case: Callable[..., Case]) -> None:
    case = build_case((6, 6), outputs=OutputControl(checkpoint_frequency=2))

    with pytest.raises(ValueError, match="needs an output directory"):
        SimulationSession(case)


def test_writers_produce_the_expected_files(tmp_path: Path, moving_case: Case) -> None:
    SimulationSession(moving_case, output_directory=str(tmp_path)).run_until_complete()

    written = sorted(entry.name for entry in tmp_path.iterdir())
    assert "Final_Total.csv" in written
    assert "Final_0.csv" in written
    assert "Original_Total.csv" in written
    assert not any(name.endswith(".partial") for name in written)


def test_a_run_folder_holds_the_one_format_the_case_selected(
    tmp_path: Path,
    build_case: Callable[..., Case],
) -> None:
    """
    The case selects one format, so no writer of another format runs, and neither the interval writes nor the first and last state leave a file of a second format behind.
    """

    case = build_case(
        (6, 6, 6),
        time=TimeControl(5, 0.25, 1),
        initial=InitialConditions(velocity=tuple((UniformValue(1.0),) for _ in range(3))),
        outputs=OutputControl(format=OutputFormat.VTK, total_frequency=2),
    )
    SimulationSession(case, output_directory=str(tmp_path)).run_until_complete()

    written = sorted(entry.name for entry in tmp_path.iterdir())
    assert written == ["2_Total.vtr", "4_Total.vtr", "Final_Total.vtr", "Original_Total.vtr"]


def test_an_interval_writes_at_the_steps_it_selects(
    tmp_path: Path,
    build_case: Callable[..., Case],
) -> None:
    """
    The interval selects the completed step count, so a frequency of 2 over five steps writes after step 2 and step 4. The first and last state go through every writer whatever the interval, and they carry the labels Original and Final instead.
    """

    case = build_case(
        (6, 6, 6),
        time=TimeControl(5, 0.25, 1),
        initial=InitialConditions(velocity=tuple((UniformValue(1.0),) for _ in range(3))),
        outputs=OutputControl(total_frequency=2),
    )
    SimulationSession(case, output_directory=str(tmp_path)).run_until_complete()

    totals = sorted(entry.name for entry in tmp_path.iterdir() if entry.name.endswith("_Total.csv"))
    assert totals == ["2_Total.csv", "4_Total.csv", "Final_Total.csv", "Original_Total.csv"]


def test_every_field_file_states_the_step_the_time_and_the_step_size(
    tmp_path: Path,
    build_case: Callable[..., Case],
) -> None:
    case = build_case(
        (6, 6),
        time=TimeControl(2, 0.25, 1),
        initial=InitialConditions(velocity=((UniformValue(1.0),), (UniformValue(0.5),))),
        outputs=OutputControl(total_frequency=1),
    )
    session = SimulationSession(case, output_directory=str(tmp_path))
    session.run_until_complete()

    first = (tmp_path / "Original_Total.csv").read_text(encoding="utf-8").splitlines()[0]
    second = (tmp_path / "2_Total.csv").read_text(encoding="utf-8").splitlines()[0]
    assert first == f"# step=0 time=0.0 dt={session.dt!r}"
    assert second == f"# step=2 time={2 * session.dt!r} dt={session.dt!r}"


def test_a_vtk_file_carries_the_time_paraview_reads(
    tmp_path: Path,
    build_case: Callable[..., Case],
) -> None:
    pyvista = pytest.importorskip("pyvista")
    case = build_case(
        (6, 6),
        time=TimeControl(2, 0.25, 1),
        initial=InitialConditions(velocity=((UniformValue(1.0),), (UniformValue(0.5),))),
        outputs=OutputControl(format=OutputFormat.VTK),
    )
    session = SimulationSession(case, output_directory=str(tmp_path))
    session.run_until_complete()

    field_data = pyvista.read(str(tmp_path / "Final_Total.vtr")).GetFieldData()
    assert field_data.GetArray("TimeValue").GetTuple1(0) == pytest.approx(session.current_time)
    assert int(field_data.GetArray("StepIndex").GetTuple1(0)) == 2
    assert field_data.GetArray("StepSize").GetTuple1(0) == pytest.approx(session.dt)


def test_write_initial_false_leaves_out_the_starting_state(
    tmp_path: Path,
    build_case: Callable[..., Case],
) -> None:
    case = build_case(
        (6, 6, 6),
        time=TimeControl(2, 0.25, 1),
        initial=InitialConditions(velocity=tuple((UniformValue(1.0),) for _ in range(3))),
    )
    SimulationSession(case, output_directory=str(tmp_path)).run_until_complete(write_initial=False)

    written = sorted(entry.name for entry in tmp_path.iterdir())
    assert not any(name.startswith("Original") for name in written)
    assert "Final_Total.csv" in written


@pytest.mark.gui
@pytest.mark.parametrize("shape,extent", [((8,), (2.0,)), ((8, 6), (2.0, 1.0)), ((8, 6, 4), (2.0, 1.0, 0.5))])
def test_vtk_output_pads_a_case_below_three_axes(
    tmp_path: Path,
    shape: tuple[int, ...],
    extent: tuple[float, ...],
) -> None:
    """
    pyevtk rejects an array that is not 3D, so a 1D or 2D case has to reach it padded to three axes.
    """

    pyvista = pytest.importorskip("pyvista")
    case = Case(
        fluid=Fluid(1.225, 0.3),
        grid=Grid(shape, extent, 1),
        time=TimeControl(1, 0.25, 1),
        initial=InitialConditions(velocity=tuple((UniformValue(1.0),) for _ in shape)),
        outputs=OutputControl(format=OutputFormat.VTK),
    )
    SimulationSession(case, output_directory=str(tmp_path)).run_until_complete()

    mesh = pyvista.read(str(tmp_path / "Final_Total.vtr"))
    assert mesh.dimensions == (*shape, *(1,) * (3 - len(shape)))
    assert mesh.bounds[: 2 * len(shape)] == pytest.approx([bound for span in extent for bound in (0.0, span)])
    assert sorted(mesh.point_data.keys()) == ["pressure", "velocity"]


@pytest.fixture(scope="session")
def serial_reference_output(
    tmp_path_factory: pytest.TempPathFactory,
    run_under_mpiexec: Callable[..., None],
) -> bytes:
    """
    The whole-domain CSV that `tests/_run_case.py` writes on one rank. Every rank count is compared against this one run, so mpiexec starts once for the reference and once for each count under test.
    """

    directory = tmp_path_factory.mktemp("serial")
    run_under_mpiexec("_run_case.py", 1, str(directory))
    return (directory / "Final_Total.csv").read_bytes()


@pytest.mark.mpi
@pytest.mark.slow
@pytest.mark.parametrize("num_procs", [2, 4, 8])
def test_the_answer_does_not_depend_on_the_rank_count(
    tmp_path: Path,
    run_under_mpiexec: Callable[..., None],
    serial_reference_output: bytes,
    num_procs: int,
) -> None:
    """
    The property the old solver did not have. Its boundary operators could not tell a global domain face from an internal partition face, so the answer moved with the decomposition.
    """

    run_under_mpiexec("_run_case.py", num_procs, str(tmp_path))
    written = (tmp_path / "Final_Total.csv").read_bytes()
    assert written == serial_reference_output, f"1 rank and {num_procs} ranks disagree"


@pytest.mark.mpi
@pytest.mark.slow
@pytest.mark.parametrize("write_procs,finish_procs", [(2, 1), (1, 4)])
def test_a_run_stopped_and_continued_answers_what_the_whole_run_answers(
    tmp_path: Path,
    run_under_mpiexec: Callable[..., None],
    serial_reference_output: bytes,
    write_procs: int,
    finish_procs: int,
) -> None:
    """
    The whole point of the checkpoint. Nothing in the file records a decomposition, so the run can also change its rank count where it stops.
    """

    run_under_mpiexec("_run_case.py", write_procs, str(tmp_path), "half")
    run_under_mpiexec("_run_case.py", finish_procs, str(tmp_path), "finish")

    written = (tmp_path / "Final_Total.csv").read_bytes()
    assert written == serial_reference_output, f"a restart at {finish_procs} ranks answers something else"
