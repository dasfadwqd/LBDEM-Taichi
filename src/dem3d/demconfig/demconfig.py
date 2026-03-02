from typing import Optional
import taichi as ti
from .types import Vector3, DomainBounds, ParticleProperties, WallProperties
from .contact_model import ContactModelConfig, LinearContactConfig, HertzContactConfig


@ti.data_oriented
class DEMSolverConfig:
    """Configuration for the DEM solver."""

    # ✅ 方案一：在类定义开头定义边界类型常量（类变量）
    BOUNDARY_TYPE_WALL = 0  # Wall boundary
    BOUNDARY_TYPE_PERIODIC = 1  # Periodic boundary

    def __init__(self,
                 domain: DomainBounds,
                 dt: float,
                 gravity: Vector3,
                 contact_model: ContactModelConfig,
                 init_particles: Optional[any] = None):
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")

        contact_model.validate()

        self.domain = domain
        self.dt = dt
        self.gravity = gravity
        self.init_particles = init_particles
        self.contact_model = contact_model

        self.particle_props = ParticleProperties()
        self.wall_props = WallProperties()

        self.domain_min, self.domain_max = domain.get_extended_bounds()
        self._setup_walls()

    def _setup_walls(self):
        self.wall_normals = [
            Vector3(1.0, 0.0, 0.0), Vector3(-1.0, 0.0, 0.0),
            Vector3(0.0, 1.0, 0.0), Vector3(0.0, -1.0, 0.0),
            Vector3(0.0, 0.0, 1.0), Vector3(0.0, 0.0, -1.0),
        ]
        self.wall_distances = [
            self.domain.xmax, -self.domain.xmin,
            self.domain.ymax, -self.domain.ymin,
            self.domain.zmax, -self.domain.zmin,
        ]


        self.boundaryType = [
            self.BOUNDARY_TYPE_WALL,  # right wall
            self.BOUNDARY_TYPE_WALL,  # left wall
            self.BOUNDARY_TYPE_WALL,  # top wall
            self.BOUNDARY_TYPE_WALL,  # bottom wall
            self.BOUNDARY_TYPE_WALL,  # front wall
            self.BOUNDARY_TYPE_WALL,  # back wall
        ]

    def set_particle_properties(self, **kwargs) -> 'DEMSolverConfig':
        for key, value in kwargs.items():
            if hasattr(self.particle_props, key):
                setattr(self.particle_props, key, value)
            else:
                raise ValueError(f"Unknown particle property: {key}")
        return self

    def set_wall_properties(self, **kwargs) -> 'DEMSolverConfig':
        for key, value in kwargs.items():
            if hasattr(self.wall_props, key):
                setattr(self.wall_props, key, value)
            else:
                raise ValueError(f"Unknown wall property: {key}")
        return self

    def set_periodic_boundaries(self, x_periodic=False, y_periodic=False, z_periodic=False):
        """
        Set periodic boundaries in batch

        Parameters:
        x_periodic: Whether the boundary in the X direction is periodic
        y_periodic: Whether the boundary in the Y direction is periodic
        z_periodic: Whether the boundary in the Z direction is periodic
        """
        if x_periodic:
            self.boundaryType[0] = self.BOUNDARY_TYPE_PERIODIC  # right
            self.boundaryType[1] = self.BOUNDARY_TYPE_PERIODIC  # left

        if y_periodic:
            self.boundaryType[2] = self.BOUNDARY_TYPE_PERIODIC  # top
            self.boundaryType[3] = self.BOUNDARY_TYPE_PERIODIC  # bottom

        if z_periodic:
            self.boundaryType[4] = self.BOUNDARY_TYPE_PERIODIC  # front
            self.boundaryType[5] = self.BOUNDARY_TYPE_PERIODIC  # back

        return self

    def update_contact_model(self, **kwargs) -> 'DEMSolverConfig':
        for key, value in kwargs.items():
            if hasattr(self.contact_model, key):
                setattr(self.contact_model, key, value)
            else:
                model_name = self.contact_model.get_model_name()
                raise ValueError(f"Unknown parameter '{key}' for {model_name} contact model")
        self.contact_model.validate()
        return self

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        model = self.contact_model
        model_name = model.get_model_name()

        summary = f"""
DEM Solver Configuration:
========================
Domain: [{self.domain.xmin}, {self.domain.xmax}] × [{self.domain.ymin}, {self.domain.ymax}] × [{self.domain.zmin}, {self.domain.zmax}]
Time step: {self.dt} s
Gravity: ({self.gravity.x}, {self.gravity.y}, {self.gravity.z}) m/s²

Particle Properties:
- Elastic modulus: {self.particle_props.elastic_modulus} Pa
- Poisson ratio: {self.particle_props.poisson_ratio}
- Density: {self.particle_props.density} kg/m³

Wall Properties:
- Density: {self.wall_props.density} kg/m³
- Elastic modulus: {self.wall_props.elastic_modulus} Pa
- Poisson ratio: {self.wall_props.poisson_ratio}

Contact Model: {model_name.upper()}
"""
        if isinstance(model, LinearContactConfig):
            summary += f"""- Normal stiffness: {model.stiffness_normal} N/m
- Tangential stiffness: {model.stiffness_tangential} N/m
- Normal damping: {model.damping_normal}
- Tangential damping: {model.damping_tangential}
"""
        elif isinstance(model, HertzContactConfig):
            summary += f"""- P-P restitution: {model.pp_restitution}
- P-W restitution: {model.pw_restitution}
"""

        summary += f"""- P-P friction: {model.pp_friction}
- P-W friction: {model.pw_friction}
- Max coordination number: {self.particle_props.max_coordinate_number}
"""

        return summary