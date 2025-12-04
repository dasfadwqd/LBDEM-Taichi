"""
Contact data structure for 3D Discrete Element Method (DEM) simulations.

Represents pairwise interactions between particles or between particles and walls,
storing geometric, material, and force/moment information required for contact
force computation and history-dependent behavior (e.g., friction).
"""

import taichi as ti

#=====================================
# Type Definitions
#=====================================

Vector3 = ti.types.vector(3, float)
Vector3i = ti.types.vector(3, int)



@ti.dataclass
class Contact:
    """Represents a contact between two bodies in the DEM simulation."""
    i: int                     # Index of the first particle or wall
    j: int                     # Index of the second particle or wall
    isActive: int              # Contact activity flag: 1 = active, 0 = inactive
    materialType_i: int        # Material type of body i
    materialType_j: int        # Material type of body j
    position: Vector3          # Global coordinates of the contact point
    force_a: Vector3           # Contact force on body i, expressed in its local frame
    moment_a: Vector3          # Contact moment on body i, in its local frame
    # Note: force_b = -force_a by Newton's third law (not stored explicitly)
    moment_b: Vector3          # Contact moment on body j, in its local frame
    shear_displacement: Vector3  # Accumulated shear displacement at the contact