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
    # 编译时常量版本（新增）
    QINV_STATIC = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17)

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
        self.nuLu = (1.0 / omega - 0.5) / 3.0
        self.nu = self.nuLu * (self.unit.dx ** 2) / self.unit.dt   # fluid kinematic viscosity [m^2/s]
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


    # =====================================
    # Map Grains to Lattice (Kernel function weight)
    # =====================================
    @ti.kernel
    def grains2lattice(self):
        """
        Optimized version with boundary-aware weight redistribution.

        When a grain is near boundary nodes, weights from excluded boundaries
        are redistributed to valid nodes, ensuring mass/momentum conservation.
        """
        # Reset fields
        self.volfrac.fill(0.0)
        self.velsolid.fill(0.0)
        self.velsum.fill(0.0)
        self.weight_sum.fill(0.0)

        V_lattice = self.unit.dx ** 3

        # =====================================
        # Grain-First Traversal
        # =====================================
        for id in range(self.dem.gf.shape[0]):
            # Pre-compute grain properties
            xc = (self.dem.gf[id].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[id].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[id].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx
            r = self.dem.gf[id].radius / self.unit.dx

            V_grain = 4.0 / 3.0 * tm.pi * self.dem.gf[id].radius ** 3
            vel_lattice = self.dem.gf[id].velocity * self.unit.dt / self.unit.dx

            # Compute bounding box
            support_radius = 1.5  # Adjust based on your kernel function

            i_min = ti.max(0, ti.cast(xc - support_radius, ti.i32))
            i_max = ti.min(self.Nx, ti.cast(xc + support_radius + 1, ti.i32))
            j_min = ti.max(0, ti.cast(yc - support_radius, ti.i32))
            j_max = ti.min(self.Ny, ti.cast(yc + support_radius + 1, ti.i32))
            k_min = ti.max(0, ti.cast(zc - support_radius, ti.i32))
            k_max = ti.min(self.Nz, ti.cast(zc + support_radius + 1, ti.i32))

            # =====================================
            # First Pass: Calculate total weights (including boundaries)
            # =====================================
            total_weight = 0.0  # Total weight including boundary nodes
            valid_weight = 0.0  # Weight only from valid nodes

            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    for k in range(k_min, k_max):
                        dist = ti.sqrt((xc - i)**2 + (yc - j)**2 + (zc - k)**2)
                        weight = self.threedelta(dist)

                        if weight < 0:
                            continue

                        total_weight += weight

                        # Check if this is a valid (non-boundary) node
                        is_boundary = self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP)
                        if not is_boundary:
                            valid_weight += weight

            # =====================================
            # Second Pass: Redistribute weights to valid nodes
            # =====================================
            if valid_weight > 0:  # Only proceed if there are valid nodes
                # Correction factor to redistribute boundary weights
                correction_factor = total_weight / valid_weight

                for i in range(i_min, i_max):
                    for j in range(j_min, j_max):
                        for k in range(k_min, k_max):
                            # Skip boundary nodes
                            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                                continue

                            dist = ti.sqrt((xc - i)**2 + (yc - j)**2 + (zc - k)**2)
                            weight_raw = self.threedelta(dist)

                            if weight_raw < 0:
                                continue

                            # Apply correction factor to redistribute boundary weights
                            weight_corrected = weight_raw * correction_factor

                            volume_contribution = weight_corrected * V_grain / V_lattice

                            # Atomic operations for thread safety
                            ti.atomic_add(self.volfrac[i, j, k], volume_contribution)
                            ti.atomic_add(self.velsum[i, j, k], weight_corrected * vel_lattice)
                            ti.atomic_add(self.weight_sum[i, j, k], weight_corrected)

        # =====================================
        # Normalize velocity field
        # =====================================
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # Clamp volume fraction
            if self.volfrac[i, j, k] > 1.0:
                self.volfrac[i, j, k] = 1.0
                print("self.volfrac[{}, {}, {}] > 1.0".format(i,j,k))

            # Compute normalized velocity
            if self.weight_sum[i, j, k] > 1e-10:
                self.velsolid[i, j, k] = self.velsum[i, j, k] / self.weight_sum[i, j, k]

    # ==========================================#
    # ----- Calculate Weight Coefficient ----- #
    # ==========================================#

    @ti.kernel
    def compute_weight(self):
        """Calculates the weighting coefficient 。

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """

        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # skip the boundary nodes
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue
            if self.volfrac[i ,j,k] > 0:
                V_lattice = self.unit.dx ** 3  # Volume of a single lattice cell
                R_lattice = ti.pow( 3 * V_lattice * self.volfrac[i ,j ,k ] / (4.0 * ti.math.pi) , 1/3)


                v_slip = (self.velsolid[i, j, k] - self.vel[i ,j ,k]) * self.unit.dx / self.unit.dt

                # Calculate the equivalent drag force
                F_d = self.compute_drag_force( 2.0 * R_lattice, v_slip, self.volfrac[i ,j ,k])
                w_d  = self.weight_coefficient(2.0 * R_lattice, v_slip, self.volfrac[i ,j ,k])

                v_slip_mag = tm.length(v_slip) * self.unit.dt / self.unit.dx
                F_d_mag = tm.length(F_d)
                F_lattice = (self.unit.rho * (self.unit.dx ** 4)) / (
                        self.unit.dt ** 2)
                F_lattice_mag = F_d_mag / F_lattice

                self.weight[i ,j ,k ] = w_d
                if self.weight[i ,j ,k ] > 1.0:
                    print("Waning! weight[{} ,{} ,{} ] > 1".format(i , j , k))

    @ti.func
    def weight_coefficient(self, dp: float, u_slip: Vector3, svf: float) -> float:
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
        dp_lattice = dp / self.unit.dx
        Wd = 3 * tm.pi * dp_lattice * self.nuLu * (1.0 - svf) * C_d

        return Wd



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
        # Cell-wise Collision
        # =====================================
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # Skip boundary cells
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            # Update fluid equilibrium based on local fluid velocity
            self.compute_feq(i, j, k)

            # Apply collision based on local solid fraction
            if self.volfrac[i, j, k] > 0.:

                self.collide_solid(i, j, k)  # solid-influenced collision
            else:
                self.collide_fluid(i, j, k)  # pure fluid BGK




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

        for q in ti.static(range(EqIMBlattice3D.Q)):
            q_inv = ti.static(EqIMBlattice3D.QINV_STATIC[q])  # ← 使用编译时常量

            Omega_s = (
                    self.f[i, j, k][q_inv]
                    - self.feq[i, j, k][q_inv]
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
            #print(self.feqsolid[i, j, k][q])

    # =====================================
    # Interpolate Fluid Forces to Grains
    # =====================================
    @ti.kernel
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
        elif 0.5 <= r <= 1.5:
            x = -3.0 * (1.0 - r) ** 2 + 1.0
            a = (5.0 - 3.0 * r - ti.sqrt(x)) / 6.0
        else:
            a = 0.0

        return a

    def initialize_complete(self):
        """完整的初始化过程"""
        # 首先初始化晶格
        self.initialize()
        # 然后映射颗粒到晶格
        self.grains2lattice()
        # 计算权重
        self.compute_weight()

    def update_coupling(self):
        """更新耦合信息"""
        # 映射颗粒到晶格
        self.grains2lattice()
        # 计算权重
        self.compute_weight()
        # 计算拖拽力
        self.lattice2grains()