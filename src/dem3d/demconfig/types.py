"""
Domain and material property definitions.
"""

from dataclasses import dataclass
import taichi as ti

Vector3 = ti.types.vector(3, float)

@dataclass
class DomainBounds:
    """Simulation domain boundaries."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def __post_init__(self):
        if self.xmin >= self.xmax:
            raise ValueError(f"xmin ({self.xmin}) must be < xmax ({self.xmax})")
        if self.ymin >= self.ymax:
            raise ValueError(f"ymin ({self.ymin}) must be < ymax ({self.ymax})")
        if self.zmin >= self.zmax:
            raise ValueError(f"zmin ({self.zmin}) must be < zmax ({self.zmax})")

    def get_extended_bounds(self):
        return (
            Vector3(self.xmin, self.ymin, self.zmin),
            Vector3(self.xmax, self.ymax, self.zmax)
        )


@dataclass
class ParticleProperties:
    """Material properties of particles."""
    elastic_modulus: float = 0  # Pa
    poisson_ratio: float = 0
    density: float = 0           # kg/m³

    max_coordinate_number: int = 0


@dataclass
class WallProperties:
    """Material properties of walls."""
    density: float = 0           # kg/m³
    elastic_modulus: float = 0    # Pa
    poisson_ratio: float = 0.0
