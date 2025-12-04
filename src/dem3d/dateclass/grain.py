"""
Grain (particle) data structure for 3D Discrete Element Method (DEM) simulations.

Represents a spherical discrete element with full 6-DOF dynamics (translation and rotation),
including kinematic state, inertial properties, and motion constraints.
"""

import taichi as ti

#=====================================
# Type Definitions
#=====================================

Vector3 = ti.types.vector(3, float)
Vector4 = ti.types.vector(4, float)
Vector3i = ti.types.vector(3, int)

Matrix3x3 = ti.types.matrix(3, 3, float)

DEMMatrix = Matrix3x3


@ti.dataclass
class Grain:
    """Represents a spherical DEM particle with full rigid-body dynamics."""
    ID: int                 # Unique particle identifier
    groupID: int            # Group identifier (e.g., for boundary or output control)
    materialType: int       # Index into material property array

    mass: float             # Particle mass
    radius: float           # Particle radius (assumed spherical)

    # Translational state (global coordinates)
    position: Vector3       # Center position
    velocity: Vector3       # Linear velocity
    acceleration: Vector3   # Linear acceleration
    force: Vector3          # Net external force

    # Rotational state (global coordinates)
    quaternion: Vector4     # Orientation as unit quaternion [w, x, y, z]
    omega: Vector3          # Angular velocity
    omega_dot: Vector3      # Angular acceleration
    inertia: DEMMatrix      # Local inertia tensor (constant for sphere, but kept general)
    moment: Vector3         # Net external moment (torque)

    # Motion constraints
    freeze: bool            # If True, particle is fixed in space (zero velocity & acceleration)
    fixvel: bool            # If True, velocity is held constant (ignores force integration)

    # Fluid force and moment
    force_fluid: Vector3
    moment_fluid: Vector3