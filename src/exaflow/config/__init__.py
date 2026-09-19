"""
Frozen value types that describe a simulation, plus the reader and writer for the input XML. Nothing in this package imports numpy, MPI or Qt, so a case can be built and compared in a test with no runtime attached.
"""

from __future__ import annotations

from .boundary_conditions import BoundaryCondition, parse_boundary_condition
from .boundaries import Boundaries, Face, FaceCondition, collect_faces
from .case import Case, SolverOptions
from .fluid import Fluid
from .grid import Grid
from .initial_conditions import FieldInitial, InitialConditions, StepValue, UniformValue
from .time_control import CheckpointFormat, OutputControl, TimeControl, parse_checkpoint_format

__all__ = [
    "Boundaries",
    "BoundaryCondition",
    "Case",
    "CheckpointFormat",
    "Face",
    "FaceCondition",
    "FieldInitial",
    "Fluid",
    "Grid",
    "InitialConditions",
    "OutputControl",
    "SolverOptions",
    "StepValue",
    "TimeControl",
    "UniformValue",
    "collect_faces",
    "parse_boundary_condition",
    "parse_checkpoint_format",
]
