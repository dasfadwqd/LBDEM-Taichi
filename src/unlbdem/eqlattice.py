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
        self.nu = nuLu * (self.unit.dx ** 2) / self.unit.dt   # fluid kinematic viscosity [m^2/s]
        self.mu = rho * self.nu

        # Solid volume fraction field (0.0 = pure fluid, 1.0 = pure solid)
        self.volfrac = ti.field(float, shape=(Nx, Ny, Nz))
        # Solid velocity field (velocity of particles in each cell)
        self.velsolid = ti.Vector.field(EqIMBlattice3D.D, float, shape=(Nx, Ny, Nz))

        # Equilibrium distribution based on solid velocity
        self.feqsolid = ti.Vector.field(EqIMBlattice3D.Q, float, shape=(Nx, Ny, Nz))

        self.weight_sum = ti.field(float,  shape=(Nx, Ny, Nz))  # Sum of weight coefficients

        self.velsum = ti.Vector.field(EqIMBlattice3D.Q, float, shape=(Nx, Ny, Nz))  # Sum of solid velocities

        # Store reference to DEM solver for data exchange
        self.dem = demslover

    # =====================================
    # Initialization Method
    # =====================================
    @ti.kernel
    def initialize(self):
        """
        Initialize the 3D lattice to equilibrium state.

        This method:
        1. Sets distribution functions to equilibrium based on initial conditions
        2. Skips boundary cells (obstacles, velocity boundaries, free-slip walls)
        3. Maps initial grain positions to the lattice
        """
        # Loop over all lattice nodes
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # Skip boundary cells (don't initialize these)
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            # Calculate equilibrium distribution functions
            self.compute_feq(i, j, k)

            # Set initial distribution functions to equilibrium state
            for q in ti.static(range(EqIMBlattice3D.Q)):
                self.f[i, j, k][q] = self.feq[i, j, k][q]

    # =====================================
    # Map Grains to Lattice (Area-Weighted)
    # =====================================
    @ti.kernel
    def grains2lattice(self):
        """
        Map DEM grain data to Eulerian lattice using area-weighted interpolation.

        Version features:
          1. Grain-first traversal: iterate grains, then affected lattice nodes.
          2. Boundary-aware normalization: weights from excluded boundary nodes
             (e.g., obstacles, velocity inlets) are redistributed to valid nodes.
        """
        # =====================================
        # Reset Accumulation Fields
        # =====================================
        self.volfrac.fill(0.0)  # Volume fraction field
        self.velsolid.fill(0.0)  # Final solid velocity field
        self.velsum.fill(0.0)  # Weighted velocity accumulator
        self.weight_sum.fill(0.0)  # Total interpolation weight per node

        V_lattice = self.unit.dx ** 3  # Volume of a single lattice cell

        # =====================================
        # Traverse All Grains
        # =====================================
        for id in ti.ndrange(self.dem.gf.shape[0]):
            # Convert grain position and radius to lattice units
            xc = (self.dem.gf[id].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[id].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[id].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx
            r = self.dem.gf[id].radius / self.unit.dx

            V_grain = 4.0 / 3.0 * tm.pi * self.dem.gf[id].radius ** 3
            vel_lattice = self.dem.gf[id].velocity * self.unit.dt / self.unit.dx

            # Determine affected lattice range (within 1.5 cells from center)
            i_min = ti.max(0, int(ti.floor(xc - 1.5)))
            i_max = ti.min(self.Nx - 1, int(ti.ceil(xc + 1.5)))
            j_min = ti.max(0, int(ti.floor(yc - 1.5)))
            j_max = ti.min(self.Ny - 1, int(ti.ceil(yc + 1.5)))
            k_min = ti.max(0, int(ti.floor(zc - 1.5)))
            k_max = ti.min(self.Nz - 1, int(ti.ceil(zc + 1.5)))

            # =====================================
            # Compute Normalization Factor
            # =====================================
            total_weight = 0.0
            valid_weight = 0.0

            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    for k in range(k_min, k_max + 1):
                        # Distance from lattice node center to grain center
                        dist = ti.sqrt((xc - i) ** 2 + (yc - j) ** 2 + (zc - k) ** 2)
                        weight = self.theredelta(dist)
                        total_weight += weight

                        # Only include nodes not marked as boundary types
                        if not (self.CT[i, j] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP)):
                            valid_weight += weight

            normalization_factor = 1.0
            if valid_weight > 1e-10:
                normalization_factor = total_weight / valid_weight

            # =====================================
            # Distribute Grain Contribution
            # =====================================
            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    for k in range(k_min, k_max + 1):
                        # Skip boundary nodes
                        if self.CT[i, j] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                            continue

                        dist = ti.sqrt((xc - i) ** 2 + (yc - j) ** 2 + (zc - k) ** 2)
                        weight = self.theredelta(dist)
                        normalized_weight = weight * normalization_factor

                        volume_contribution = normalized_weight * V_grain / V_lattice

                        ti.atomic_add(self.volfrac[i, j, k], volume_contribution)
                        ti.atomic_add(self.weight_sum[i, j, k], normalized_weight)
                        ti.atomic_add(self.velsum[i, j, k], normalized_weight * vel_lattice)

        # =====================================
        # Finalize Solid Velocity Field
        # =====================================
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # Compute weighted average velocity
            if self.weight_sum[i, j, k] > 1e-10:
                self.velsolid[i, j, k] = self.velsum[i, j, k] / self.weight_sum[i, j, k]

            # Enforce physical bounds on volume fraction
            if self.volfrac[i, j, k] > 1.0:
                print(f"WARNING at ({i},{j},{k}): Volume fraction = {self.volfrac[i, j, k]}")
                self.volfrac[i, j, k] = 1.0  # Clamp to maximum physical value


    # =====================================
    # Drag Force: Tenneti Model
    # =====================================
    @ti.func
    def compute_drag_force(self, dp: float, u_slip: Vector3, svf: float) -> Vector3:
        """
        Compute drag force on a particle using the Tenneti drag model for dense suspensions.

        This model incorporates solid volume fraction (ε_p = svf) to account for
        particle-particle interactions in dense granular flows.

        **Drag Force:**
            F_d = 3π d_p μ₀ (1 - ε_p) C_d(Re_p, ε_p) u_slip

        **Drag Coefficient C_d:**
            C_d = (1 - ε_p) [ C_d0 / (1 - ε_p)³ + A(ε_p) + B(Re_p, ε_p) ]

            - C_d0 = 1 + 0.15·Re_p^0.687
            - A(ε_p) = 5.81·ε_p / (1 - ε_p)³ + 0.48·ε_p^(1/3) / (1 - ε_p)⁴
            - B(Re_p, ε_p) = ε_p³·Re_p·[0.95 + 0.61·ε_p³ / (1 - ε_p)²]

        **Reynolds Number:**
            Re_p = (1 - ε_p)·ρ_f·d_p·|u_slip| / μ₀

        Args:
            dp (float): Particle diameter [m]
            u_slip (Vector3): Fluid-particle relative velocity [m/s]
            svf (float): Solid volume fraction ε_p ∈ [0, 1)

        Returns:
            Vector3: Drag force vector [N]

        Reference:
            Tenneti et al. (2011), Int. J. Multiphase Flow, 37(9), 1072–1092.
            DOI: 10.1016/j.ijmultiphaseflow.2011.05.010
        """

        # =====================================
        # Relative Velocity Magnitude
        # =====================================
        u_slip_mag = tm.length(u_slip)

        # =====================================
        # Fluid Properties & Reynolds Number
        # =====================================
        rho_f = self.rho0  # Fluid density [kg/m³]
        mu0 = self.mu  # Dynamic viscosity [Pa·s]

        # Particle Reynolds number (corrected by (1 - svf))
        Re_p = (1.0 - svf) * rho_f * dp * u_slip_mag / mu0

        # =====================================
        # Drag Coefficient C_d
        # =====================================
        C_d = 0.0
        if 1.0 - svf > 1e-9:  # Avoid singularity as svf → 1
            # Single-particle baseline
            Cd0 = 1.0 + 0.15 * tm.pow(Re_p, 0.687)

            # Static correction term A(ε_p)
            A_eps = (5.81 * svf / ((1.0 - svf) ** 3) +
                     0.48 * tm.pow(svf, 1.0 / 3.0) / ((1.0 - svf) ** 4))

            # Dynamic correction term B(Re_p, ε_p)
            svf3 = svf ** 3
            B_eps = svf3 * Re_p * (0.95 + 0.61 * svf3 / ((1.0 - svf) ** 2))

            # Assemble total drag coefficient
            C_d = (1.0 - svf) * (Cd0 / ((1.0 - svf) ** 3) + A_eps + B_eps)

        # =====================================
        # Drag Force Vector
        # =====================================
        F_drag = 3.0 * tm.pi * dp * mu0 * (1.0 - svf) * C_d * u_slip

        return F_drag

    # =====================================
    # the delta function
    # =====================================
    @ti.func
    def theredelta(self, r) -> float:
        a = 0.0
        if r < 0.5:
            x = -3.0 * r ** 2 + 1.0
            a = (1.0 + ti.sqrt(x)) / 3.0
        elif r <= 1.5:
            x = -3.0 * (1.0 - r) ** 2 + 1.0
            a = (5.0 - 3.0 * r - ti.sqrt(x)) / 6.0

        return a