'''
Contact model parameter settings
'''
from dataclasses import dataclass
from abc import ABC, abstractmethod

import taichi as ti
Vector3 = ti.types.vector(3, float)

class ContactModelConfig(ABC):
    """Base class for contact models."""

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def validate(self):
        pass


@dataclass
class LinearContactConfig(ContactModelConfig):
    """Linear spring-dashpot contact model."""
    stiffness_normal: float
    stiffness_tangential: float
    damping_normal: float
    damping_tangential: float
    pp_friction: float
    pw_friction: float


    def get_model_name(self) -> str:
        return "linear"

    def validate(self):
        if self.stiffness_normal <= 0 or self.stiffness_tangential <= 0:
            raise ValueError("Stiffness values must be positive")
        if self.damping_normal < 0 or self.damping_tangential < 0:
            raise ValueError("Damping values must be non-negative")
        if not (0 <= self.pp_friction <= 1) or not (0 <= self.pw_friction <= 1):
            raise ValueError("Friction coefficients must be between 0 and 1")


@dataclass
class HertzContactConfig(ContactModelConfig):
    """Hertz-Mindlin contact model."""
    pp_friction: float
    pw_friction: float
    pp_restitution: float
    pw_restitution: float


    def get_model_name(self) -> str:
        return "hertz"

    def validate(self):
        if not (0 <= self.pp_friction <= 1) or not (0 <= self.pw_friction <= 1):
            raise ValueError("Friction coefficients must be between 0 and 1")
        if not (0 <= self.pp_restitution <= 1) or not (0 <= self.pw_restitution <= 1):
            raise ValueError("Restitution coefficients must be between 0 and 1")