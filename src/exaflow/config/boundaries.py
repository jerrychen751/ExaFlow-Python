from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .boundary_conditions import BoundaryCondition


class Face(Enum):
    """
    One of the six faces of the global domain. The value is (axis, side): axis 0 is x, 1 is y and 2 is z; side 0 is the low end of that axis and side 1 is the high end. A face whose axis is at or above the case dimension does not exist for that case.
    """

    LEFT = (0, 0)
    RIGHT = (0, 1)
    TOP = (1, 0)
    BOTTOM = (1, 1)
    FRONT = (2, 0)
    BACK = (2, 1)

    @property
    def axis(self) -> int:
        return self.value[0]

    @property
    def is_low(self) -> bool:
        return self.value[1] == 0

    @property
    def opposite(self) -> Face:
        return Face((self.axis, 1 - self.value[1]))


def collect_faces(dimension: int) -> tuple[Face, ...]:
    """
    Return the faces that exist for a case of this dimension, low end before high end, x before y before z.
    """

    return tuple(face for face in Face if face.axis < dimension)


@dataclass(frozen=True, slots=True)
class FaceCondition:
    """
    The boundary condition on one face, together with the values that condition needs.
    """

    kind: BoundaryCondition = BoundaryCondition.NO_SLIP
    velocity: tuple[float, ...] = ()  # prescribed components in axis order; read only when kind is INFLOW, and then as long as the case dimension
    pressure: float = 0.0  # prescribed value; read only when kind is OUTFLOW

    def __post_init__(self) -> None:
        if self.kind == BoundaryCondition.TIME_DEPENDENT:
            raise NotImplementedError("Time-dependent boundary conditions are not implemented.")
        if self.kind == BoundaryCondition.INFLOW and not self.velocity:
            raise ValueError("An INFLOW face needs a velocity tuple.")


@dataclass(frozen=True, slots=True)
class Boundaries:
    """
    The condition on each of the six domain faces. A periodic face must be paired with a periodic face opposite it, because the ghost exchange wraps the two together.
    """

    left: FaceCondition = FaceCondition()
    right: FaceCondition = FaceCondition()
    top: FaceCondition = FaceCondition()
    bottom: FaceCondition = FaceCondition()
    front: FaceCondition = FaceCondition()
    back: FaceCondition = FaceCondition()

    def __post_init__(self) -> None:
        for face in (Face.LEFT, Face.TOP, Face.FRONT):
            here = self.find_face(face).kind == BoundaryCondition.PERIODIC
            there = self.find_face(face.opposite).kind == BoundaryCondition.PERIODIC
            if here != there:
                raise ValueError(
                    f"Periodic faces must be paired; got {face.name}={self.find_face(face).kind.value!r}, "
                    f"{face.opposite.name}={self.find_face(face.opposite).kind.value!r}."
                )

    def find_face(self, face: Face) -> FaceCondition:
        return getattr(self, face.name.lower())

    def is_periodic(self, axis: int) -> bool:
        """
        Report whether the pair of faces on this axis wraps around.
        """

        low = next(f for f in Face if f.axis == axis and f.is_low)
        return self.find_face(low).kind == BoundaryCondition.PERIODIC
