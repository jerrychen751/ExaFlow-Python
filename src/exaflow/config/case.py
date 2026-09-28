from __future__ import annotations

from dataclasses import dataclass, field

from .boundary_conditions import BoundaryCondition
from .boundaries import Boundaries, collect_faces
from .fluid import Fluid
from .grid import Grid
from .initial_conditions import InitialConditions
from .time_control import OutputControl, TimeControl


@dataclass(frozen=True, slots=True)
class SolverOptions:
    """
    Which physical terms the spatial operator assembles, and the discretization chosen for each. A term switched off contributes nothing to the right-hand side.

    One scheme is implemented per term: "Upwind" for convection and "CentralDifference" for the viscous term. Any other name is rejected here rather than accepted and ignored, so the scheme this records is always the scheme that runs.
    """

    include_convection: bool = True
    include_diffusion: bool = True
    include_pressure: bool = False
    convection_scheme: str = "Upwind"
    viscous_scheme: str = "CentralDifference"

    def __post_init__(self) -> None:
        if self.convection_scheme != "Upwind":
            raise NotImplementedError(
                f"convection_scheme must be 'Upwind'; {self.convection_scheme!r} is not implemented."
            )
        if self.viscous_scheme != "CentralDifference":
            raise NotImplementedError(
                f"viscous_scheme must be 'CentralDifference'; {self.viscous_scheme!r} is not implemented."
            )


@dataclass(frozen=True, slots=True)
class Case:
    """
    One complete problem definition: the fluid, the grid, how long to march, what happens on each face, where the fields start, which terms to solve and how often to write. Every part is frozen, so a case can be built once, compared, and shared between ranks without a copy.

    A case holds no MPI communicator and no output directory. Those belong to a run, not to the problem, and are given to the solver instead.
    """

    fluid: Fluid
    grid: Grid
    time: TimeControl
    boundaries: Boundaries = field(default_factory=Boundaries)
    initial: InitialConditions = field(default_factory=InitialConditions)
    solver: SolverOptions = field(default_factory=SolverOptions)
    outputs: OutputControl = field(default_factory=OutputControl)

    def __post_init__(self) -> None:
        dimension = self.grid.dimension
        for face in collect_faces(dimension):
            condition = self.boundaries.find_face(face)
            if condition.kind != BoundaryCondition.INFLOW:
                continue
            if len(condition.velocity) != dimension:
                raise ValueError(
                    f"{face.name} is an INFLOW face, so it needs {dimension} velocity components, "
                    f"got {condition.velocity!r}."
                )
        if self.initial.velocity and len(self.initial.velocity) != dimension:
            raise ValueError(
                f"initial.velocity needs one entry per axis ({dimension}), got {len(self.initial.velocity)}."
            )

    @property
    def dimension(self) -> int:
        return self.grid.dimension
