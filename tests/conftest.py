"""
Fixtures every test module shares: the paths this repository ships, the two factories that build a case and a block, and the launcher the MPI tests drive mpiexec with.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pytest

from exaflow.config import Case, Fluid, Grid, TimeControl
from exaflow.mpi.process_grid import ProcessGrid
from exaflow.mpi.subdomain import Subdomain

if TYPE_CHECKING:
    from PySide6 import QtWidgets

TESTS_DIRECTORY = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Skip every test marked `mpi` when mpiexec is absent from PATH, so a machine with no launcher reports skips instead of failures.
    """

    if shutil.which("mpiexec") is not None:
        return
    skip_mpi = pytest.mark.skip(reason="mpiexec is not on PATH")
    for item in items:
        if "mpi" in item.keywords:
            item.add_marker(skip_mpi)


@pytest.fixture(scope="session")
def template_case_path() -> Path:
    """
    The absolute path of the input XML this repository ships, so a test reads it whatever directory pytest starts in.
    """

    return TESTS_DIRECTORY.parent / "examples" / "input_template.xml"


@pytest.fixture(scope="session")
def qt_application() -> QtWidgets.QApplication:
    """
    The one Qt application object a process is allowed, built for widgets rather than for a message loop alone. Every GUI test module takes this fixture, because a module that builds a bare QCoreApplication first leaves every later widget test with an object it cannot use. Qt reads QT_QPA_PLATFORM when the object is built, so this sets the offscreen platform first and needs no display.
    """

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert isinstance(application, QtWidgets.QApplication)
    return application


@pytest.fixture(scope="session")
def build_case() -> Callable[..., Case]:
    """
    Return a factory that builds a valid Case of the given shape. The extent is 1.0 meter per axis and the spacing therefore follows the point count. Pass any Case field by keyword to replace the default.
    """

    def make(shape: tuple[int, ...] = (8, 8, 8), **overrides: object) -> Case:
        defaults: dict[str, object] = dict(
            fluid=Fluid(1.225, 0.3),
            grid=Grid(shape, tuple(1.0 for _ in shape), 1),
            time=TimeControl(4, 0.25, 1),
        )
        defaults.update(overrides)
        return Case(**defaults)  # type: ignore[arg-type]

    return make


@pytest.fixture(scope="session")
def build_subdomain() -> Callable[..., Subdomain]:
    """
    Return a factory that builds the block one rank owns. `counts` defaults to one rank per axis, which gives the whole grid to rank 0.
    """

    def make(grid: Grid, counts: tuple[int, ...] | None = None, rank: int = 0) -> Subdomain:
        return Subdomain(grid, ProcessGrid(counts or (1,) * grid.dimension), rank)

    return make


@pytest.fixture(scope="session")
def run_under_mpiexec() -> Callable[..., None]:
    """
    Return a launcher that runs one helper script in `tests/` under mpiexec at the given rank count and fails the test when it exits non-zero. The helper reports a wrong answer through its exit status, so the launcher needs no output contract.
    """

    def run(script: str, num_procs: int, *arguments: str) -> None:
        command = ["mpiexec", "-n", str(num_procs), sys.executable, str(TESTS_DIRECTORY / script), *arguments]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
        assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]

    return run
