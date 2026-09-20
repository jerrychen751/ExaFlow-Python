from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class Fluid:
    """
    Constant properties of the working fluid.
    """

    rho: float  # density in kg/m^3; finite and positive
    nu: float  # kinematic viscosity in m^2/s; zero is allowed and makes the viscous term vanish

    def __post_init__(self) -> None:
        if not math.isfinite(self.rho) or self.rho <= 0.0:
            raise ValueError(f"rho must be finite and > 0, got {self.rho}.")
        if not math.isfinite(self.nu) or self.nu < 0.0:
            raise ValueError(f"nu must be finite and >= 0, got {self.nu}.")
