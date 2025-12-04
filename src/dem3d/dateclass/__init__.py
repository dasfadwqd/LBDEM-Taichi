"""
Core data structures for TaichiLB's 3D Discrete Element Method (DEM) simulations.

This module aggregates fundamental DEM entities used to model particle-scale dynamics,
including:

- Particles (Grain): mass, radius, position, velocity, orientation, etc.
- Walls: planar boundaries defined by normal and distance.
- Materials: physical properties such as density, stiffness, friction, and damping.
- Contacts: interaction records between particle–particle or particle–wall pairs,
  including contact geometry, forces, and history-dependent variables.
"""

from .contact import Contact
from .wall import Wall
from .grain import Grain
from .material import Material

__all__ = [
    "Contact",
    "Material",
    "Wall",
    "Grain",
]