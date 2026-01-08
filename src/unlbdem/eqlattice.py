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
        self.weight = ti.field(float, shape=(Nx, Ny, Nz))  # weighting factor

        # Equilibrium distribution based on solid velocity
        self.feqsolid = ti.Vector.field(EqIMBlattice3D.Q, float, shape=(Nx, Ny, Nz))

        self.weight_sum = ti.field(float,  shape=(Nx, Ny, Nz))  # Sum of weight coefficients

        self.velsum = ti.Vector.field(EqIMBlattice3D.D, float, shape=(Nx, Ny, Nz))  # Sum of solid velocities

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

        # map grains to lattice in case the initial state needs to be saved
        self.grains2lattice()

    # =====================================
    # Map Grains to Lattice (Kernel function weight)
    # =====================================
    @ti.func
    def grains2lattice(self):
        """
        Map DEM grain data to Eulerian lattice using Kernel function weight interpolation.

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
        self.weight.fill(0.0)

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
                        weight = self.threedelta(dist)
                        total_weight += weight

                        # Only include nodes not marked as boundary types
                        if not (self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP)):
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
                        if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                            continue

                        dist = ti.sqrt((xc - i) ** 2 + (yc - j) ** 2 + (zc - k) ** 2)
                        weight = self.threedelta(dist)
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
            # calculate the weighting coefficient
            self.compute_weight(i, j, k)

    # ==========================================#
    # ----- Calculate Weight Coefficient ----- #
    # ==========================================#

    @ti.func
    def compute_weight(self, i: int, j: int, k: int):
        """Calculates the weighting coefficient 。

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """

        V_lattice = self.unit.dx ** 3  # Volume of a single lattice cell
        R_lattice = ti.pow( 3 * V_lattice * self.volfrac[i ,j ,k ] / (4.0 * ti.math.pi) , 1 / 3)
        vel_lattice = self.velsolid[i, j, k] * self.unit.dx / self.unit.dt

        v_slip = vel_lattice - self.vel[i ,j ,k] * self.unit.dx / self.unit.dt

        # Calculate the equivalent drag force
        F_lattice = self.compute_drag_force( 2.0 * R_lattice, v_slip, self.volfrac[i ,j ,k ])
        v_slip_mag = tm.length(v_slip)
        F_lattice_mag = tm.length(F_lattice)
        self.weight[i ,j ,k ] = F_lattice_mag / ( v_slip_mag * self.rho0)

    # =====================================
    # Collision Step:  Fluid-Solid
    # =====================================
    @ti.kernel
    def collide(self):
        """
        Perform weighted collision operator based on local solid volume fraction.

        The post-collision distribution is:
            fpc = f + B·Ω_s + (1 - B)·Ω_f

        where:
            Ω_s = f_inv - feq_inv + feq_solid - f   (solid relaxation)
            Ω_f = -ω·(f - feq)                      (fluid BGK relaxation)
            B   = weight[i, j, k]                   (interpolation weight)

        Steps:
          1. Map DEM grains to lattice (volfrac, velsolid).
          2. For each fluid/solid cell, apply appropriate collision.
          3. Accumulate hydrodynamic forces on grains.
        """
        # =====================================
        # Map Grains to Lattice
        # =====================================
        self.grains2lattice()

        # =====================================
        # Cell-wise Collision
        # =====================================
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # Skip boundary cells
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            # Update fluid equilibrium based on local fluid velocity
            self.compute_feq(i, j, k)

            # Apply collision based on local solid fraction
            if self.volfrac[i, j, k] > 0.0:
                self.collide_solid(i, j, k)  # solid-influenced collision
            else:
                self.collide_fluid(i, j, k)  # pure fluid BGK

        # =====================================
        # Compute Hydrodynamic Forces on Grains
        # =====================================
        self.lattice2grains()

    # =====================================
    # Fluid-Only Collision (BGK)
    # =====================================
    @ti.func
    def collide_fluid(self, i: int, j: int, k: int):
        """
        Standard BGK collision for pure fluid cells.

        fpc = (1 - ω)·f + ω·feq

        Args:
            i, j, k (int): Lattice indices.
        """
        for q in ti.static(range(EqIMBlattice3D.Q)):
            self.fpc[i, j, k][q] = (
                    (1.0 - self.omega[i, j, k]) * self.f[i, j, k][q]
                    + self.omega[i, j, k] * self.feq[i, j, k][q]
            )

    # =====================================
    # Solid-Influenced Collision
    # =====================================
    @ti.func
    def collide_solid(self, i: int, j: int, k: int):
        """
        Weighted collision operator for cells with solid presence.

        Combines fluid relaxation (Ω_f) and solid momentum exchange (Ω_s).

        Args:
            i, j, k (int): Lattice indices.
        """
        # Update equilibrium using solid velocity
        self.compute_feq_solid(i, j, k)

        # Apply weighted collision
        for q in ti.static(range(EqIMBlattice3D.Q)):
            # Solid momentum exchange term
            Omega_s = (
                    self.f[i, j, k][EqIMBlattice3D.qinv[q]]
                    - self.feq[i, j, k][EqIMBlattice3D.qinv[q]]
                    + self.feqsolid[i, j, k][q]
                    - self.f[i, j, k][q]
            )
            # Fluid BGK relaxation term
            Omega_f = -self.omega[i, j, k] * (self.f[i, j, k][q] - self.feq[i, j, k][q])

            # Weighted combination
            self.fpc[i, j, k][q] = (
                    self.f[i, j, k][q]
                    + self.weight[i, j, k] * Omega_s
                    + (1.0 - self.weight[i, j, k]) * Omega_f
            )

    # =====================================
    # Solid-Based Equilibrium (feq_solid)
    # =====================================
    @ti.func
    def compute_feq_solid(self, i: int, j: int, k: int):
        """
        Compute equilibrium distribution using solid velocity.

        feq = w·ρ·[1 + 3(c·u) + 4.5(c·u)² - 1.5|u|²]

        Note: Uses precomputed lattice weights and speeds (D3Q19 assumed).

        Args:
            i, j, k (int): Lattice indices.
        """
        u = self.velsolid[i, j, k]
        uv = tm.dot(u, u)  # |u|²

        for q in ti.static(range(EqIMBlattice3D.Q)):
            cu = tm.dot(EqIMBlattice3D.c[q], u)
            self.feqsolid[i, j, k][q] = EqIMBlattice3D.w[q] * self.rho[i, j, k] * (
                    1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uv
            )

    # =====================================
    # Interpolate Fluid Forces to Grains
    # =====================================
    @ti.func
    def lattice2grains(self):
        """
        Interpolate hydrodynamic force from lattice to DEM grains.

        Steps:
          1. Reset fluid force on all grains.
          2. For each grain, gather weighted fluid velocity and volfrac
             from surrounding lattice nodes (within 2-cell radius).
          3. Compute slip velocity and drag force via Tenneti model.
          4. Accumulate force on grain.
        """
        self.dem.gf.force_fluid.fill(0.0)


        for id in ti.ndrange(self.dem.gf.shape[0]):
            # Convert grain position to lattice coordinates
            xc = (self.dem.gf[id].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[id].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[id].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx

            # Define local search window (±2 cells)
            x_begin = ti.max(0, int(xc - 2))
            x_end = ti.min(self.Nx, int(xc + 2))
            y_begin = ti.max(0, int(yc - 2))
            y_end = ti.min(self.Ny, int(yc + 2))
            z_begin = ti.max(0, int(zc - 2))
            z_end = ti.min(self.Nz, int(zc + 2))

            fluid_vel_particle = Vector3(0.0, 0.0, 0.0)
            volfrac_particle = 0.0

            # Interpolate fluid state from neighboring nodes
            for i in range(x_begin, x_end):
                for j in range(y_begin, y_end):
                    for k in range(z_begin, z_end):
                        # Skip boundary nodes
                        if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                            continue

                        dist = ti.sqrt((xc - i) ** 2 + (yc - j) ** 2 + (zc - k) ** 2)
                        weight = self.threedelta(dist)  # interpolation kernel

                        fluid_vel_particle += self.vel[i, j, k] * weight
                        volfrac_particle += self.volfrac[i, j, k] * weight

            # Compute drag force if particle interacts with fluid
            if volfrac_particle > 1e-10:
                eps_p = volfrac_particle  # solid volume fraction
                d_p = 2.0 * self.dem.gf[id].radius  # particle diameter
                fluid_vel = fluid_vel_particle * self.unit.dx / self.unit.dt  # back to physical units
                u_slip = self.dem.gf[id].velocity - fluid_vel

                F_d = self.compute_drag_force(d_p, u_slip, eps_p)
                self.dem.gf[id].force_fluid += F_d


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
            u_slip (Vector3): u_p - u_f Fluid-particle relative velocity [m/s]
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
        F_drag = - 3.0 * tm.pi * dp * mu0 * (1.0 - svf) * C_d * u_slip

        return F_drag

    # =====================================
    # the delta function
    # =====================================
    @ti.func
    def threedelta(self, r) -> float:
        a = 0.0
        if r < 0.5:
            x = -3.0 * r ** 2 + 1.0
            a = (1.0 + ti.sqrt(x)) / 3.0
        elif 0.5 < r <= 1.5:
            x = -3.0 * (1.0 - r) ** 2 + 1.0
            a = (5.0 - 3.0 * r - ti.sqrt(x)) / 6.0
        else:
            a = 0.0

        return a