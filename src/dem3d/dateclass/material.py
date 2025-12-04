"""
Material data structure for 3D Discrete Element Method (DEM) simulations.

Encapsulates physical and contact-related properties used in both linear and Hertzian
contact models, including elasticity, damping, friction, and restitution.
"""

import taichi as ti




@ti.dataclass
class Material:
    """Material properties for DEM particles and walls."""
    density: float             # Mass density (kg/m³)
    elasticModulus: float      # Young's modulus (Pa)
    poissonRatio: float        # Poisson's ratio (dimensionless)

    # Friction and restitution
    coefficientFriction: float      # Coulomb friction coefficient
    coefficientRestitution: float   # Coefficient of restitution (0–1)

    # Hertz-Mindlin contact stiffness
    stiffness_normal: float         # Normal contact stiffness
    stiffness_tangent: float        # Tangential (shear) contact stiffness

    # Linear spring-dashpot model damping ratios
    dp_nratio: float                # Normal critical damping ratio
    dp_sratio: float                # Shear critical damping ratio