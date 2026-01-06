'''
This program adopts the particle equivalent method proposed in the research paper by
Professor Limin Wang, implementing the un-resolved LBM-DEM coupling simulation.
Reference: DOI: https://doi.org/10.1016/j.cej.2023.142898
'''

import taichi as ti
import taichi.math as tm


# =====================================
# Module Imports
# =====================================
# LBM module - Lattice Boltzmann Method components
from src.lbm3d.lbm_solver3d import BasicLattice3D
from src.lbm3d.lbmutils import CellType

# DEM module - Discrete Element Method components
from src.dem3d.demsolver import DEMSolver

from src.unlbdem.utils import Interpolation


# =====================================
# Type Definitions
# =====================================
Vector3 = ti.types.vector(3, float)
#=====================================
# Unresolved LBM-DEM Coupling Class
# =====================================
class EqIMBlattice3D(BasicLattice3D):
    """
    Extend the BasicLattice3D class with unresolved LBM-DEM coupling calculation module.

    The unresolved coupling method is characterized by grid size larger than particle size,
    which brings high computational efficiency. The unresolved method adopted in this class
    is modified based on the traditional Immersed Boundary Method (IMB): particles are treated
    as equivalent entities (particle equivalence treatment), and the governing equations of IMB
    are adjusted to account for the effect of equivalent particles on the fluid phase. Notably,
    the effect of the fluid phase on particles is still based on the Tenetti empirical formula.

    Key Features:
        1. LBM (Lattice Boltzmann Method): Used for numerical simulation of the fluid phase;
        2. DEM (Discrete Element Method): Used for dynamic simulation of the particle phase;
        3. IMB (Immersed Boundary Method): The traditional method modified in this coupling strategy;
        4. Tenetti empirical formula: Applied to calculate the fluid-particle interaction (fluid-to-particle effect);
        5. Unresolved coupling: A coupling mode where grid size > particle size (opposite to resolved coupling).


       Args:
           Nx (int): Number of lattice nodes in x-direction
           Ny (int): Number of lattice nodes in y-direction
           Nz (int): Number of lattice nodes in z-direction
           omega (float): Relaxation frequency (related to fluid viscosity)
           dx (float): Lattice spacing [m]
           dt (float): LBM time step [s]
           rho (float): Fluid density [kg/m³]
           demslover (DEMSolver): DEM solver instance for particle dynamics
       """

    def __init__(self, Nx: int, Ny: int, Nz: int, omega: float,
                 dx: float, dt: float, rho: float, demslover: DEMSolver):
        """
        Initialize the 3D unresolved LBM-DEM lattice.

        This constructor sets up the lattice structure and initializes fields
        required for fluid-particle coupling.
        """
        # Initialize parent class (BasicLattice3D)
        super().__init__(Nx, Ny, Nz, omega, dx, dt, rho)

        # =====================================
        # Fluid-Particle Coupling Fields
        # =====================================

        # Particle identification field
        self.rho0 = rho  # fluid density
        nuLu = (1.0 / omega - 0.5) / 3.0
        self.nu = nuLu * (self.unit.dx ** 2) / self.unit.dt
        self.mu = rho * self.nu

        # Solid volume fraction field (0.0 = pure fluid, 1.0 = pure solid)
        self.volfrac = ti.field(float, shape=(Nx, Ny, Nz))
        # Solid velocity field (velocity of particles in each cell)
        self.velsolid = ti.Vector.field(EqIMBlattice3D.D, float, shape=(Nx, Ny, Nz))

        # Equilibrium distribution based on solid velocity
        self.feqsolid = ti.Vector.field(EqIMBlattice3D.Q, float, shape=(Nx, Ny, Nz))

        self.temp_weights = ti.field(float, shape=(self.Nx, self.Ny, self.Nz))

        # Store reference to DEM solver for data exchange
        self.dem = demslover

