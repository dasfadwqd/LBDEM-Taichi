"""
Unresolved LBM-DEM Method for 3D Fluid-Particle Coupling

This module implements the unresolved Lattice Boltzmann Method (LBM) coupled with
Discrete Element Method (DEM) for simulating fluid-particle interactions in 3D.

The unresolved approach treats particles smaller than the lattice spacing by using
volume-averaged Navier-Stokes equations.

"""

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
class Unresolvedlattice3D(BasicLattice3D):
    """
    3D Lattice Boltzmann solver with unresolved DEM coupling.

    This class extends BasicLattice3D to implement volume-averaged Navier-Stokes
    equations for fluid-particle interaction. The "unresolved" approach is used
    when particle sizes are smaller than the lattice spacing, requiring volume
    averaging of fluid equations.

    Key Features:
    - Volume fraction tracking for solid particles
    - Hydrodynamic force calculation on particles
    - Modified equilibrium distribution for particle-laden flows
    - Effective viscosity correction based on solid volume fraction

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
        self.rho0 = rho # fluid density
        nuLu = (1.0/ omega - 0.5)/ 3.0
        self.nu = nuLu * (self.unit.dx ** 2) / self.unit.dt
        self.mu = rho * self.nu


        # Solid volume fraction field (0.0 = pure fluid, 1.0 = pure solid)
        self.volfrac = ti.field(float, shape=(Nx, Ny, Nz))

        self.Fq = ti.Vector.field(BasicLattice3D.Q, float,
                                   shape=(Nx, Ny, Nz))  # equilibrium force

        # Solid velocity field (velocity of particles in each cell)
        self.velsolid = ti.Vector.field(Unresolvedlattice3D.D, float, shape=(Nx, Ny, Nz))

        # Equilibrium distribution based on solid velocity
        self.feqsolid = ti.Vector.field(Unresolvedlattice3D.Q, float, shape=(Nx, Ny, Nz))

        # Hydrodynamic force field (force exerted by fluid on particles)
        self.hydroforce = ti.Vector.field(Unresolvedlattice3D.D, float, shape=(Nx, Ny, Nz))
        self.temp_weights = ti.field(float, shape=(self.Nx, self.Ny, self.Nz))

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
            for q in ti.static(range(Unresolvedlattice3D.Q)):
                self.f[i, j, k][q] = self.feq[i, j, k][q]



    # ===========================================#
    # ----- Data Exchange from DEM to LBM ----- #
    # ===========================================#

    @ti.kernel
    def grains2lattice(self):
        """
        Map DEM grain information to LBM lattice grid using kernel-based interpolation.
        """
        # Reset solid volume fraction to zero (pure fluid state)
        self.volfrac.fill(0.0)

        # Process each grain independently
        for grain_id in range(self.dem.gf.shape[0]):
            # 获取颗粒位置
            xc = (self.dem.gf[grain_id].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[grain_id].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[grain_id].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx

            r = self.dem.gf[grain_id].radius / self.unit.dx

            kernel_support = 1.5
            i_min = ti.max(0, int(ti.floor(xc - kernel_support)))
            i_max = ti.min(self.Nx - 1, int(ti.ceil(xc + kernel_support)))
            j_min = ti.max(0, int(ti.floor(yc - kernel_support)))
            j_max = ti.min(self.Ny - 1, int(ti.ceil(yc + kernel_support)))
            k_min = ti.max(0, int(ti.floor(zc - kernel_support)))
            k_max = ti.min(self.Nz - 1, int(ti.ceil(zc + kernel_support)))

            # First pass: calculate weights
            valid_weight_sum = 0.0
            boundary_weight_sum = 0.0


            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    for k in range(k_min, k_max + 1):
                        dist = ti.sqrt((xc - i) ** 2 + (yc - j) ** 2 + (zc - k) ** 2)
                        if dist <= kernel_support:
                            w = self.threedelta(dist)
                            self.temp_weights[i, j, k] = w
                            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                                boundary_weight_sum += w
                            else:
                                valid_weight_sum += w

            # Calculate correction factor
            total_weight = valid_weight_sum + boundary_weight_sum
            correction_factor = 1.0
            if valid_weight_sum > 1e-10:
                correction_factor = total_weight / valid_weight_sum

            # Second pass: update volume fraction
            V_grain = 4.0 / 3.0 * tm.pi * self.dem.gf[grain_id].radius ** 3
            V_cell = self.unit.dx ** 3

            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    for k in range(k_min, k_max + 1):
                        if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                            continue

                        w_raw = self.temp_weights[i, j, k]
                        if w_raw > 1e-10:
                            w_corrected = w_raw * correction_factor
                            # 原子操作，避免竞争条件
                            ti.atomic_add(self.volfrac[i, j, k], w_corrected * V_grain / V_cell)


    # ==============================#
    # ----- Collision Process -----#
    # ==============================#

    @ti.kernel
    def collide(self):
        """BGK collision operator in LBM.
                fpc = (1-omega)*f + omega*feq + Fq
        """
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # skip the boundary nodes
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            # update the equilibrium state

            self.compute_omega(i , j, k)
            self.compute_feq(i, j, k)

            self.compute_Fq(i,j,k)

            # collision (relax the distribution functions towards equilibrium)
            for q in ti.static(range(BasicLattice3D.Q)):


                # relax the distribution functions
                self.fpc[i, j, k][q] = (
                        (1.0 - self.omega[i, j, k]) * self.f[i, j, k][q] +
                        self.omega[i, j, k] * self.feq[i, j, k][q] +
                        self.Fq[i ,j ,k][q]
                )


    # =====================================
    # Equilibrium Distribution Function
    # =====================================
    @ti.func
    def compute_feq(self, i: int, j: int, k: int):
        """
        Calculate equilibrium distribution functions for particle-laden flow.

        The equilibrium distribution is modified to account for solid volume fraction:
        feq = w_q * ρ * [1 + 3(c_q·u) + 4.5(c_q·u)²/fvf - 1.5|u|²/fvf]

        Where:
        - w_q: Lattice weight for direction q
        - ρ: Fluid density [kg/m³]
        - c_q: Lattice velocity vector in direction q [lattice units]
        - u: Macroscopic fluid velocity vector [m/s]
        - fvf: Fluid volume fraction (1 - solid volume fraction ε_p)
        - cs: Speed of sound (1/√3 in lattice units, embedded in coefficients)

        Note: This is a purely local operation dependent on cell (i,j,k) properties.

        Args:
            i (int): Lattice index in x-direction
            j (int): Lattice index in y-direction
            k (int): Lattice index in z-direction
        """
        # Squared magnitude of macroscopic velocity (|u|²)
        u_sq = tm.dot(self.vel[i, j, k], self.vel[i, j, k])

        # Fluid volume fraction (fvf = 1 - solid volume fraction)
        fvf = 1.0 - self.volfrac[i, j, k]

        # Safety check: prevent non-positive fluid volume fraction
        if fvf <= 1e-10:  # Using small epsilon instead of 0 to avoid numerical issues
            print("Warning! Fluid Volume Fraction <= 0.0 at cell ({}, {}, {})".format(i, j, k))

        # Calculate equilibrium distribution for all D3Q19 directions
        for q in ti.static(range(BasicLattice3D.Q)):
            # Dot product of lattice velocity and macroscopic velocity (c_q·u)
            c_dot_u = tm.dot(BasicLattice3D.c[q], self.vel[i, j, k])

            # Modified equilibrium distribution with volume fraction correction
            self.feq[i, j, k][q] = BasicLattice3D.w[q] * self.rho[i, j, k] * (
                    1.0 + 3.0 * c_dot_u * fvf + 4.5 * (c_dot_u ** 2) * fvf - 1.5 * u_sq * fvf
            )

    @ti.func
    def compute_Fq(self, i: int, j: int, k: int):
        """
        Calculate force contribution to distribution functions (F_q) for particle-laden flow.

        This term accounts for the momentum exchange between particles and fluid,
        modified by the fluid volume fraction (fvf) to handle dense suspensions.

        Formula:
        F_q = w_q * (1 - 0.5ω) * [3c_q - 3u/fvf + 9(c_q·u)c_q/fvf]

        Where:
        - w_q: Lattice weight for direction q
        - ω: Relaxation frequency
        - c_q: Lattice velocity vector in direction q
        - u: Macroscopic fluid velocity vector
        - fvf: Fluid volume fraction (1 - solid volume fraction)
        """
        # Fluid volume fraction (fvf = 1 - solid volume fraction)
        fvf = 1.0 - self.volfrac[i, j, k]
        if fvf <= 1e-10:
            print("Warning! Fluid Volume Fraction <= 0.0 at cell ({}, {}, {})".format(i, j, k))

        F = self.hydroforce[i, j, k]  # Vector3, force density [N/m³]

        for q in ti.static(range(BasicLattice3D.Q)):
            c = BasicLattice3D.c[q]
            w = BasicLattice3D.w[q]
            u = self.vel[i, j, k]
            omega = self.omega[i, j, k]

            term = (
                    3.0 * c
                    - 3.0 * u
                    + 9.0 * tm.dot(c, u) * c
            )

            # 标量：F_q = w * (1 - ω/2) * (term · F)
            F_q = w * (1.0 - 0.5 * omega) * tm.dot(term, F)

            self.Fq[i, j, k][q] = F_q

    # =====================================
    # Effective Viscosity Calculation
    # =====================================
    @ti.func
    def compute_omega(self, i: int, j: int, k: int):
        """
        Calculate effective relaxation frequency accounting for particle presence.

        The effective kinematic viscosity is modified by solid volume fraction (svf)
        using an empirical correlation for dense suspensions:

        ν_eff = ν * [1 + 1.25 * svf * (1-svf)/0.64]²

        Where:
        - ν: Base kinematic viscosity of pure fluid
        - svf: Solid volume fraction (ε_p)

        The relaxation frequency is then updated using the relationship:
        ω = 1 / (3ν_eff + 0.5)

        Args:
            i (int): Lattice index in x-direction
            j (int): Lattice index in y-direction
            k (int): Lattice index in z-direction
        """
        # Calculate base kinematic viscosity from current relaxation frequency
        # Relationship: ν = (1/ω - 0.5) / 3

        nuLu = self.nu * self.unit.dt / (self.unit.dx ** 2)

        # Solid volume fraction at current cell
        svf = self.volfrac[i, j, k]

        # Calculate effective viscosity with empirical particle correction
        # Accounts for increased flow resistance in particle-laden regions
        nu_eff = nuLu * (1.0 + 1.25 * svf * (1.0 - svf) / 0.64) ** 2  # Simplified 2.5/2 to 1.25

        # Update relaxation frequency based on effective viscosity
        self.omega[i, j, k] = 1.0 / (3.0 * nu_eff + 0.5)

    @ti.kernel
    def stream(self):
        """Streaming process in LBM.

            Update the fluid density and velocity after the boundary conditions have been applied.
        """
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            # copy distribution function at rest
            self.f[i, j, k][0] = self.fpc[i, j, k][0]

            # propogate from the neighbours to the current cell
            for q in range(1, BasicLattice3D.Q):
                # index of the neighbouring cell
                iNext, jNext, kNext = self.compute_neigh_index(q, i, j, k)

                # streaming
                if self.CT[iNext, jNext, kNext] & (
                        CellType.FLUID | CellType.VEL_ZOUHE | CellType.VEL_EXIT | CellType.Pre_ZOUHE):  # propogation
                    self.f[i, j, k][q] = self.fpc[iNext, jNext, kNext][q]
                elif self.CT[iNext, jNext, kNext] & CellType.OBSTACLE:  # bounce-back
                    self.f[i, j, k][q] = self.fpc[i, j, k][BasicLattice3D.qinv[q]]
                elif self.CT[iNext, jNext, kNext] & CellType.VEL_LADD:  # modified bounce-back
                    cu = tm.dot(BasicLattice3D.c[q], self.vel[iNext, jNext, kNext])
                    self.f[i, j, k][q] = self.fpc[i, j, k][BasicLattice3D.qinv[q]] + 2.0 * BasicLattice3D.w[q] * \
                                         self.rho[i, j, k] * cu / BasicLattice3D.cssq
                elif self.CT[iNext, jNext, kNext] & CellType.FREE_SLIP:  # specular reflection
                    if self.CT[iNext, jNext, kNext] & (CellType.LEFT | CellType.RIGHT):
                        self.f[i, j, k][q] = self.fpc[i, jNext, kNext][BasicLattice3D.qsyx[q]]
                    elif self.CT[iNext, jNext, kNext] & (CellType.BOTTOM | CellType.TOP):
                        self.f[i, j, k][q] = self.fpc[iNext, j, kNext][BasicLattice3D.qsyy[q]]
                    elif self.CT[iNext, jNext, kNext] & (CellType.BACK | CellType.FRONT):
                        self.f[i, j, k][q] = self.fpc[iNext, jNext, k][BasicLattice3D.qsyz[q]]

            # update density and velocity if fluid
            if self.CT[i, j, k] & CellType.FLUID: self.compute_rhof_vel(i, j, k)

            # apply boundary conditions (wet node approaches)
            for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
                if self.CT[i, j, k] & CellType.VEL_ZOUHE:
                    self.vel_zouHe(i, j, k)
                elif self.CT[i, j, k] & CellType.VEL_EXIT:
                    self.bc_vel_exit(i, j, k)
                elif self.CT[i, j, k] & CellType.Pre_ZOUHE:
                    self.pre_zouHe(i, j, k)


    # ==============================================#
    # ----- Compute Fluid Density & Velocity ----- #
    # ==============================================#
    @ti.func
    def compute_rhof_vel(self, i: int, j: int, k: int):
        """Update density and velocity using conservation laws with external force."""
        self.rho[i, j, k] = 0.0
        self.vel[i, j, k] = Vector3(0.0, 0.0, 0.0)

        # Sum over all directions to get momentum
        for q in range(BasicLattice3D.Q):
            self.rho[i, j, k] += self.f[i, j, k][q]
            self.vel[i, j, k] += BasicLattice3D.c[q] * self.f[i, j, k][q]

        # Check for positive density
        assert self.rho[i, j, k] > 0, f"Density at ({i}, {j}, {k}) is non-positive!"

        # Normalize to get velocity
        self.vel[i, j, k] /= self.rho[i, j, k]

        # Add half-step external force contribution (from formula)
        # Note: hydroforce is force density [N/m³], so we need to convert to acceleration
        # Formula: u = (momentum / rho) + (dt / 2 / rho) * f_ext

        self.vel[i, j, k] += 0.5 *  self.hydroforce[i, j, k] / self.rho[i, j, k]

        # Optional: check velocity magnitude
        if tm.dot(self.vel[i, j, k], self.vel[i, j, k]) ** 0.5 > BasicLattice3D.velmax:
            print(f"Warning: Velocity too large at ({i}, {j}, {k})")


    # =====================================
    # Hydrodynamic force Calculation
    # =====================================
    @ti.kernel
    def compute_hydrodynamic_force(self):
        self.dem.gf.force_fluid.fill(0.0)
        for id in range(self.dem.gf.shape[0]):
            # convert grain position and size to lattice units
            xc = (self.dem.gf[id].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[id].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[id].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx
            r = self.dem.gf[id].radius / self.unit.dx

            # extents of the lattice covered by the grain
            x_begin = max(int(xc - r - 2), 0)
            x_end = min(int(xc + r + 2), self.Nx)
            y_begin = max(int(yc - r - 2), 0)
            y_end = min(int(yc + r + 2), self.Ny)
            z_begin = max(int(zc - r - 2), 0)
            z_end = min(int(zc + r + 2), self.Nz)

            svf = 0.0
            u_f = Vector3(0.0, 0.0, 0.0)

            for i in range(x_begin, x_end):
                for j in range(y_begin, y_end):
                    for k in range(z_begin, z_end):
                        dist = ti.sqrt((xc - i) ** 2 + (yc - j) ** 2 + (zc - k) ** 2)
                        if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                            continue
                        if self.volfrac[i, j, k] > 0.0:
                            w = self.threedelta(dist)
                            svf += self.volfrac[i, j, k] * w
                            u_f += self.vel[i, j, k] * w
            # compute the drag force
            F_d = self.compute_drag_force(id , svf, u_f)
            # compute the lift force
            #F_l = self.compute_lift_force(id)
            # accumulate the hydrodynamic force to dem
            self.dem.gf[id].force_fluid += F_d


    # =====================================
    # Hydrodynamic Force Coupling (DEM to LBM)
    # =====================================
    # =====================================
    # Grain-to-Lattice Mapping (DEM → LBM)
    # =====================================
    @ti.kernel
    def force2lattice(self):
        """
        Map DEM grain volume and hydrodynamic force to LBM lattice in a single pass.
        Uses threedelta kernel (support radius 1.5 lattice units) for interpolation.
        Boundary weight correction ensures conservation of particle volume and force
        in fluid cells near obstacles.
        """
        # Reset solid volume fraction and hydrodynamic force to pure fluid state
        self.volfrac.fill(0.0)
        self.hydroforce.fill(0.0)

        # Process each grain independently
        for grain_id in range(self.dem.gf.shape[0]):
            # Convert grain position [m] to lattice coordinates (floating-point)
            xc = (self.dem.gf[grain_id].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[grain_id].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[grain_id].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx

            r_lat = self.dem.gf[grain_id].radius / self.unit.dx
            kernel_support = 1.5

            i_min = ti.max(0, int(ti.floor(xc - kernel_support)))
            i_max = ti.min(self.Nx - 1, int(ti.ceil(xc + kernel_support)))
            j_min = ti.max(0, int(ti.floor(yc - kernel_support)))
            j_max = ti.min(self.Ny - 1, int(ti.ceil(yc + kernel_support)))
            k_min = ti.max(0, int(ti.floor(zc - kernel_support)))
            k_max = ti.min(self.Nz - 1, int(ti.ceil(zc + kernel_support)))

            # First pass: calculate total interpolation weights for correction
            valid_weight_sum = 0.0
            boundary_weight_sum = 0.0

            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    for k in range(k_min, k_max + 1):
                        dist = ti.sqrt((xc - i) ** 2 + (yc - j) ** 2 + (zc - k) ** 2)
                        if dist <= kernel_support:
                            w_raw = self.threedelta(dist)
                            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                                boundary_weight_sum += w_raw
                            else:
                                valid_weight_sum += w_raw

            total_weight = valid_weight_sum + boundary_weight_sum
            correction_factor = 1.0
            if valid_weight_sum > 1e-10:
                correction_factor = total_weight / valid_weight_sum

            # Second pass: update both volume fraction and hydrodynamic force
            V_grain = 4.0 / 3.0 * tm.pi * self.dem.gf[grain_id].radius ** 3
            V_cell = self.unit.dx ** 3
            fluid_force = self.dem.gf[grain_id].force_fluid

            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    for k in range(k_min, k_max + 1):
                        if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                            continue
                        dist = ti.sqrt((xc - i) ** 2 + (yc - j) ** 2 + (zc - k) ** 2)
                        if dist <= kernel_support:
                            w_raw = self.threedelta(dist)
                            if w_raw > 1e-10:
                                w_corrected = w_raw * correction_factor

                                # Accumulate solid volume fraction (dimensionless)
                                ti.atomic_add(self.volfrac[i, j, k], w_corrected * V_grain / V_cell)

                                # Accumulate hydrodynamic force source term with unit scaling
                                ti.atomic_add(self.hydroforce[i, j, k],
                                              w_corrected * fluid_force * self.unit.dt ** 2 / (
                                                          self.unit.rho * self.unit.dx))


    # =====================================
    # Drag Force Calculation
    # =====================================
    @ti.func
    def compute_drag_force(self, id: int, svf: float, u_f: Vector3) -> Vector3:
        """
        Calculate the drag force on a grain using Ergun-Wen-Yu drag model.

        This drag model accounts for particle-particle interactions through the
        solid volume fraction ε_p (svf), making it suitable for dense particle suspensions.

        **Drag Force Formula:**
        F_d = 3π d_p μ₀ (1 - ε_p) C_d(Re_p, ε_p) (u_f - u_p)

        Where:
        - d_p: Particle diameter [m]
        - μ₀: Dynamic viscosity of fluid [Pa·s]
        - ε_p: Solid volume fraction (svf, local particle concentration)
        - C_d: Drag coefficient (function of particle Reynolds number Re_p and ε_p)
        - u_f: Fluid velocity at particle location [m/s]
        - u_p: Particle velocity [m/s]

        **Drag Coefficient C_d:**
        C_d(Re_p, ε_p) = (1 - ε_p) [C_d0/(1-ε_p)³ + A(ε_p) + B(Re_p, ε_p)]

        Where:
        - C_d0 = 1 + 0.15·Re_p^0.687  (single particle drag coefficient)
        - A(ε_p) = 5.81ε_p/(1-ε_p)³ + 0.48ε_p^(1/3)/(1-ε_p)⁴  (static correction term)
        - B(Re_p, ε_p) = ε_p³·Re_p·[0.95 + 0.61ε_p³/(1-ε_p)²]  (dynamic correction term)
        - Re_p: Particle Reynolds number = (1-ε_p)·ρ_f·d_p·|u_f-u_p| / μ₀

        Args:
            id (int): Index of the grain in DEM solver
            svf (float): Solid volume fraction (ε_p)
            u_f (Vector3): Fluid velocity at particle location [m/s]

        Returns:
            Vector3: Drag force vector [N] in 3D

        References:
            - S. Tenneti, R. Garg, and S. Subramaniam (2011)
              doi: 10.1016/j.ijmultiphaseflow.2011.05.010
        """
        # =====================================
        # Step 1: Get Particle Properties
        # =====================================
        # Particle diameter [m]
        d_p = 2.0 * self.dem.gf[id].radius

        # Particle velocity [m/s]
        u_p = self.dem.gf[id].velocity

        # =====================================
        # Step 2: Calculate Relative Velocity
        # =====================================
        u_rel = self.unit.dx * u_f / self.unit.dt - u_p  # Slip velocity between fluid and particle
        u_rel_mag = tm.length(u_rel)  # Magnitude of slip velocity

        # =====================================
        # Step 3: Fluid Properties & Reynolds Number
        # =====================================
        rho_f = self.rho0  # Fluid density [kg/m³]
        mu0 = self.mu  # Fluid dynamic viscosity [Pa·s]

        # Particle Reynolds number (accounting for solid volume fraction)
        Re_p = (1.0 - svf) * rho_f * d_p * u_rel_mag / mu0

        # =====================================
        # Step 4: Calculate Drag Coefficient C_d
        # =====================================
        C_d = 0.0
        if 1.0 - svf > 1e-8:  # Avoid division by near-zero (dilute limit protection)
            # Single particle drag coefficient
            Cd0 = 1.0 + 0.15 * Re_p ** 0.687

            # Static correction term (A)
            A_eps = (5.81 * svf / ((1.0 - svf) ** 3) +
                     0.48 * (svf ** (1.0 / 3.0)) / ((1.0 - svf) ** 4))

            # Dynamic correction term (B)
            B_eps = (svf ** 3) * Re_p * (0.95 +
                                         0.61 * (svf ** 3) / ((1.0 - svf) ** 2))

            # Total drag coefficient
            C_d = (1.0 - svf) * (Cd0 / ((1.0 - svf) ** 3) + A_eps + B_eps)

        # =====================================
        # Step 5: Calculate Drag Force
        # =====================================
        F_drag = 3.0 * tm.pi * d_p * mu0 * (1.0 - svf) * C_d * u_rel

        return F_drag

    @ti.func
    def threedelta(self, r) -> float:
        a = 0.0
        if r < 0.5:
            x = -3.0 * r ** 2 + 1.0
            a = (1.0 + x * 0.5) / 3.0
        elif r <= 1.5:
            x = -3.0 * (1.0 - r) ** 2 + 1.0
            a = (5.0 - 3.0 * r - ti.sqrt(x)) / 6.0
        return a

    '''
    @ti.func
    def compute_lift_force(self, grain_id: int) -> Vector3:
        """
        Calculate the Saffman lift force on a grain due to velocity gradient.
    
        The lift force arises when a particle moves through a non-uniform flow field
        (shear flow). The force is perpendicular to both the relative velocity and
        the vorticity vector, pushing the particle toward regions of lower velocity.
    
        **Saffman Lift Force Formula:**
    
        F_l = 1.61 * d_p² * √(ρμ/|ω|) * [(u - v) × ω]
    
        Where:
        - d_p: Particle diameter [m]
        - ρ: Fluid density [kg/m³]
        - μ: Dynamic viscosity [Pa·s]
        - u: Fluid velocity at particle location [m/s]
        - v: Particle velocity [m/s]
        - ω = ∇×u: Vorticity (curl of velocity field) [1/s]
        - |ω|: Magnitude of vorticity
        - ×: Cross product
    
        **Physical Interpretation:**
        - Magnus effect analog: particle in shear experiences lateral force
        - Direction: perpendicular to both slip velocity and vorticity
        - Magnitude: proportional to √(shear rate) and slip velocity
        - Important in: inertial microfluidics, particle separation, turbulence
    
        **Assumptions:**
        - Small particle Reynolds number (Re_p << 1)
        - Particle size << flow length scale
        - Weak shear (particle rotation negligible)
        - Valid for Stokes flow regime
    
        Args:
            grain_id (int): Index of the grain in DEM solver
        Returns:
            Vector3: Lift force vector [N] in 3D
        """
        # =====================================
        # Step 1: Get Particle Properties
        # =====================================
        # Particle position in physical coordinates [m]
        x_p = self.dem.gf[grain_id].position
    
        # Particle velocity [m/s]
        v_p = self.dem.gf[grain_id].velocity
    
        # Particle diameter [m]
        d_p = 2.0 * self.dem.gf[grain_id].radius
    
        # =====================================
        # Step 2: Interpolate Fluid Properties at Particle Location
        # =====================================
        # Convert particle position to lattice coordinates
        xc = (x_p[0] - self.dem.config.xmin + 0.5 * self.unit.dx) / self.unit.dx
        yc = (x_p[1] - self.dem.config.ymin + 0.5 * self.unit.dx) / self.unit.dx
        zc = (x_p[2] - self.dem.config.zmin + 0.5 * self.unit.dx) / self.unit.dx
    
        # Get integer indices of the cell containing the particle
        i = int(ti.floor(xc))
        j = int(ti.floor(yc))
        k = int(ti.floor(zc))
    
        # Ensure indices are within bounds
        i = ti.max(1, ti.min(self.Nx - 2, i))
        j = ti.max(1, ti.min(self.Ny - 2, j))
        k = ti.max(1, ti.min(self.Nz - 2, k))
    
        # Interpolate fluid velocity at particle location using trilinear interpolation
        # (For simplicity, we use the cell center velocity here)
        # More accurate: use trilinear interpolation from 8 surrounding cells
        u_fluid = self.vel[i, j, k]
    
        # =====================================
        # Step 3: Calculate Vorticity (∇×u)
        # =====================================
        # Vorticity components using central difference scheme:
        # ω_x = ∂u_z/∂y - ∂u_y/∂z
        # ω_y = ∂u_x/∂z - ∂u_z/∂x
        # ω_z = ∂u_y/∂x - ∂u_x/∂y
    
        # Get velocity components at neighboring cells for finite difference
        # Note: dx_lattice = 1.0 in lattice units
    
        # ω_x component
        duzdj = (self.vel[i, j + 1, k][2] - self.vel[i, j - 1, k][2]) / (2.0 * self.unit.dx)
        duydk = (self.vel[i, j, k + 1][1] - self.vel[i, j, k - 1][1]) / (2.0 * self.unit.dx)
        omega_x = duzdj - duydk
    
        # ω_y component
        duxdk = (self.vel[i, j, k + 1][0] - self.vel[i, j, k - 1][0]) / (2.0 * self.unit.dx)
        duzdi = (self.vel[i + 1, j, k][2] - self.vel[i - 1, j, k][2]) / (2.0 * self.unit.dx)
        omega_y = duxdk - duzdi
    
        # ω_z component
        duydi = (self.vel[i + 1, j, k][1] - self.vel[i - 1, j, k][1]) / (2.0 * self.unit.dx)
        duxdj = (self.vel[i, j + 1, k][0] - self.vel[i, j - 1, k][0]) / (2.0 * self.unit.dx)
        omega_z = duydi - duxdj
    
        # Vorticity vector
        omega = Vector3([omega_x, omega_y, omega_z])
    
        # Magnitude of vorticity |ω|
        omega_mag = tm.length(omega)
    
        # =====================================
        # Step 4: Calculate Relative Velocity
        # =====================================
        # Slip velocity: fluid velocity - particle velocity [m/s]
        u_rel = u_fluid - v_p
    
        # =====================================
        # Step 5: Calculate Lift Force
        # =====================================
        # Initialize lift force to zero
        F_lift = Vector3([0.0, 0.0, 0.0])
    
        # Check if vorticity is significant (avoid division by zero)
        if omega_mag > 1e-10:
            # Fluid properties
            rho = self.rho[i, j, k]  # Fluid density [kg/m³]
    
            # Calculate dynamic viscosity from kinematic viscosity
            # ν = (1/ω - 0.5) / 3 in lattice units
            # Convert to physical units: ν_phys = ν_lattice * dx² / dt
            nu_lattice = (1.0 / self.omega[i, j, k] - 0.5) / 3.0
            nu_phys = nu_lattice * (self.unit.dx ** 2) / self.unit.dt
            mu = rho * nu_phys  # Dynamic viscosity [Pa·s]
    
            # Saffman coefficient: K = 1.61 * d_p² * √(ρμ/|ω|)
            K_saffman = 1.61 * (d_p ** 2) * ti.sqrt(rho * mu / omega_mag)
    
            # Cross product: (u - v) × ω
            # This gives the direction perpendicular to both slip velocity and vorticity
            cross_product = tm.cross(u_rel, omega)
    
            # Saffman lift force [N]
            F_lift = K_saffman * cross_product
    
        # =====================================
        # Step 6: Return Lift Force
        # =====================================
        return F_lift
    '''

