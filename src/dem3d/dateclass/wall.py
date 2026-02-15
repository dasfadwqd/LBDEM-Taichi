"""
Wall data structure for 3D Discrete Element Method (DEM) simulations.

Represents an infinite planar boundary using the plane equation:  
n · x = d, where n is the unit normal vector and d is the distance from the origin.
Used for particle–wall contact detection and interaction.
"""

import taichi as ti

#=====================================
# Type Definitions
#=====================================

Vector3 = ti.types.vector(3, float)


@ti.dataclass
class Wall:
    """Represents an infinite planar wall in 3D space."""
    normal: Vector3      # Unit normal vector pointing *into* the valid simulation domain
    distance: float      # Signed distance from the origin to the wall (i.e., d in n·x = d)
    materialType: int    # Material identifier for contact response (e.g., friction, stiffness)
    boundaryType: int    # Boundary type ( Wall boundary = 0, Periodic boundary = 1 )