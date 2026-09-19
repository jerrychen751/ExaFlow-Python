from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


@dataclass(frozen=True, slots=True)
class TimeControl:
    """
    How far the run marches and how each step is taken. `num_steps` is the step budget of the whole run, counted from time zero, so a run restored at step 400 of 1000 takes 600 more steps. `cfl` scales the advective step size limit. `integration_order` selects the explicit Runge-Kutta scheme: 1 is Euler, 2 is the midpoint method and 3 is the Shu-Osher TVD scheme.

    `end_time` is the simulated time in seconds the run marches to, or None to march until the step budget runs out. With an end time set, `num_steps` becomes the cap that stops a run whose step size shrinks faster than the time left. `adaptive_time_step` recomputes the step size from the current state before every step instead of once from the initial state, which costs one reduction across the ranks per step.
    """

    num_steps: int
    cfl: float
    integration_order: int = 1
    end_time: float | None = None
    adaptive_time_step: bool = False

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {self.num_steps}.")
        if not math.isfinite(self.cfl) or self.cfl <= 0.0:
            raise ValueError(f"cfl must be finite and > 0, got {self.cfl}.")
        if self.integration_order not in (1, 2, 3):
            raise ValueError(f"integration_order must be 1, 2 or 3, got {self.integration_order}.")
        if self.end_time is not None and (not math.isfinite(self.end_time) or self.end_time <= 0.0):
            raise ValueError(f"end_time must be finite and > 0, got {self.end_time}.")


class CheckpointFormat(str, Enum):
    """
    The file format used for restart checkpoints. Values stay strings so case XML remains readable.
    """

    CSV = "CSV"
    VTK = "VTK"


def parse_checkpoint_format(value: str) -> CheckpointFormat:
    """
    Parse a checkpoint format from a string. An unknown value raises instead of silently selecting another format.
    """

    try:
        return CheckpointFormat(value)
    except ValueError as exc:
        allowed = ", ".join(fmt.value for fmt in CheckpointFormat)
        raise ValueError(f"Unknown checkpoint format {value!r}. Allowed: {allowed}.") from exc


@dataclass(frozen=True, slots=True)
class OutputControl:
    """
    How often a run streams intermediate states and saves restart checkpoints, counted in completed time steps.

    `checkpoint_format` selects readable CSV or binary VTK for restart files. `stream_frequency` is the interval sent to a connected viewer; -1 sends the starting and final states without intermediate states. `checkpoint_frequency` is the interval between restart files; -1 writes no checkpoint, including at the end.
    """

    checkpoint_format: CheckpointFormat = CheckpointFormat.CSV
    stream_frequency: int = -1
    checkpoint_frequency: int = -1

    def __post_init__(self) -> None:
        for name in ("stream_frequency", "checkpoint_frequency"):
            frequency = getattr(self, name)
            if frequency != -1 and frequency < 1:
                raise ValueError(f"{name} must be -1 or >= 1, got {frequency}.")

    def is_due(self, frequency: int, step: int) -> bool:
        """
        Report whether a writer whose interval is `frequency` writes at this step. The step is the count of completed steps, so a frequency of 2 over five steps selects steps 2 and 4.
        """

        return frequency >= 1 and step % frequency == 0
