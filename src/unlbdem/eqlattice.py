"""
Unresolved LBM-DEM coupling simulation based on the particle equivalence method
proposed by Prof. Limin Wang (DOI: https://doi.org/10.1016/j.cej.2023.142898).
This implementation combines Lattice Boltzmann Method (LBM) for fluid dynamics
and Discrete Element Method (DEM) for particle motion, using a modified
Immersed Boundary Method (IBM) with unresolved grid resolution (dx > particle size).
"""

import taichi as ti
import taichi.math as tm

# =====================================
# Module Imports
# =====================================

# LBM components
from src.lbm3d.lbm_solver3d import BasicLattice3D
from src.lbm3d.lbmutils import CellType

# DEM components
from src.dem3d.demsolver import DEMSolver

# Coupling utilities
from src.unlbdem.utils import Interpolation


# =====================================
# Type Definitions
# =====================================

Vector3 = ti.types.vector(3, float)


# =====================================
# Unresolved LBM-DEM Coupling Class
# =====================================

class EqIMBlattice3D(BasicLattice3D):
    """
    Extends BasicLattice3D to support unresolved LBM-DEM coupling via particle equivalence.

    In unresolved coupling, the lattice spacing exceeds particle diameter, enabling
    high computational efficiency. Particles are treated as equivalent continuum entities,
    modifying the Immersed Boundary Method (IBM). Fluid-to-particle forces use the
    Tenneti drag model, while particle-to-fluid coupling adjusts equilibrium distributions.

    Args:
        Nx (int): Number of lattice nodes along x.
        Ny (int): Number of lattice nodes along y.
        Nz (int): Number of lattice nodes along z.
        omega (float): Relaxation frequency (related to viscosity).
        dx (float): Lattice spacing [m].
        dt (float): Time step [s].
        rho (float): Fluid density [kg/m³].
        demslover (DEMSolver): DEM solver instance for particle dynamics.
    """

    # Static inverse direction mapping for D3Q19 lattice
    QINV_STATIC = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17)

    def __init__(self, Nx: int, Ny: int, Nz: int, omega: float,
                 dx: float, dt: float, rho: float, demslover: DEMSolver):
        """Initialize the unresolved LBM-DEM lattice."""
        super().__init__(Nx, Ny, Nz, omega, dx, dt, rho)

        # Fluid properties
        self.rho0 = rho  # fluid density
        self.nuLu = (1.0 / omega - 0.5) / 3.0  # kinematic viscosity in lattice units
        self.nu = self.nuLu * (self.unit.dx ** 2) / self.unit.dt  # [m²/s]
        self.mu = rho * self.nu  # dynamic viscosity [Pa·s]
        self.omega0 = omega
        self.SmagorinskyConstant = 0.1  # Smagorinsky constant (typically 0.1-0.2)
        self.cssq = 1.0/3.0 # speed of sound squared

        # Coupling fields
        self.volfrac = ti.field(float, shape=(Nx, Ny, Nz))          # solid volume fraction
        self.velsolid = ti.Vector.field(self.D, float, shape=(Nx, Ny, Nz))  # solid velocity
        self.weight = ti.field(float, shape=(Nx, Ny, Nz))           # interpolation weight
        self.feqsolid = ti.Vector.field(self.Q, float, shape=(Nx, Ny, Nz))  # solid-based feq
        self.weight_sum = ti.field(float, shape=(Nx, Ny, Nz))       # sum of weights
        self.velsum = ti.Vector.field(self.D, float, shape=(Nx, Ny, Nz))    # weighted velocity sum

        # Reference to DEM solver
        self.dem = demslover

    # =====================================
    # Initialization
    # =====================================
    @ti.kernel
    def initialize(self):
        """Initialize lattice to equilibrium state, skipping boundary cells."""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue
            self.compute_feq(i, j, k)
            for q in ti.static(range(self.Q)):
                self.f[i, j, k][q] = self.feq[i, j, k][q]

    # =====================================
    # Map DEM Grains to Lattice
    # =====================================

    @ti.kernel
    def grains2lattice(self):
        """
        Map particle data to lattice using kernel interpolation with boundary-aware redistribution.

        Ensures conservation by redistributing weights from excluded boundary nodes to valid ones.
        """
        self.volfrac.fill(0.0)
        self.velsolid.fill(0.0)
        self.velsum.fill(0.0)
        self.weight_sum.fill(0.0)

        V_lattice = self.unit.dx ** 3

        for id in range(self.dem.gf.shape[0]):
            # Convert particle position to lattice coordinates
            xc = (self.dem.gf[id].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[id].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[id].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx
            r = self.dem.gf[id].radius / self.unit.dx
            V_grain = 4.0 / 3.0 * tm.pi * self.dem.gf[id].radius ** 3
            vel_lattice = self.dem.gf[id].velocity * self.unit.dt / self.unit.dx

            support_radius = 1.5
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
                        if not (self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP)):
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
                            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                                continue
                            dist = ti.sqrt((xc - i)**2 + (yc - j)**2 + (zc - k)**2)
                            weight_raw = self.threedelta(dist)
                            if weight_raw < 0:
                                continue
                            weight_corrected = weight_raw * correction_factor
                            volume_contribution = weight_corrected * V_grain / V_lattice
                            ti.atomic_add(self.volfrac[i, j, k], volume_contribution)
                            ti.atomic_add(self.velsum[i, j, k], weight_corrected * vel_lattice)
                            ti.atomic_add(self.weight_sum[i, j, k], weight_corrected)

        # =====================================
        # Normalize velocity field
        # =====================================
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.volfrac[i, j, k] > 1.0:
                self.volfrac[i, j, k] = 1.0
                print("Warning: volfrac[{}, {}, {}] > 1.0".format(i, j, k))
            if self.weight_sum[i, j, k] > 1e-10:
                self.velsolid[i, j, k] = self.velsum[i, j, k] / self.weight_sum[i, j, k]

    # =====================================
    # Compute Weight Coefficient
    # =====================================

    @ti.kernel
    def compute_weight(self):
        """Compute local weight coefficient based on drag force model."""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue
            if self.volfrac[i, j, k] > 0:
                V_lattice = self.unit.dx ** 3
                R_lattice = ti.pow(3 * V_lattice * self.volfrac[i, j, k] / (4.0 * tm.pi), 1.0 / 3.0)
                v_slip = (self.velsolid[i, j, k] - self.vel[i, j, k]) * self.unit.dx / self.unit.dt
                w_d = self.weight_coefficient(2.0 * R_lattice, v_slip, self.volfrac[i, j, k])
                self.weight[i, j, k] = w_d
                if w_d > 1.0:
                    print("Warning: weight[{}, {}, {}] > 1.0".format(i, j, k))

    @ti.func
    def weight_coefficient(self, dp: float, u_slip: Vector3, svf: float) -> float:
        """
        Compute dimensionless weight coefficient using the Tenneti drag model.

        Returns a lattice-scaled drag coefficient used in the collision operator.
        """
        u_slip_mag = tm.length(u_slip)
        rho_f = self.rho0
        mu0 = self.mu
        Re_p = (1.0 - svf) * rho_f * dp * u_slip_mag / mu0

        C_d = 0.0
        if 1.0 - svf > 1e-9:
            Cd0 = 1.0 + 0.15 * tm.pow(Re_p, 0.687)
            A_eps = (5.81 * svf / ((1.0 - svf) ** 3) +
                     0.48 * tm.pow(svf, 1.0 / 3.0) / ((1.0 - svf) ** 4))
            svf3 = svf ** 3
            B_eps = svf3 * Re_p * (0.95 + 0.61 * svf3 / ((1.0 - svf) ** 2))
            C_d = (1.0 - svf) * (Cd0 / ((1.0 - svf) ** 3) + A_eps + B_eps)

        dp_lattice = dp / self.unit.dx
        Wd = 3.0 * tm.pi * dp_lattice * self.nuLu * (1.0 - svf) * C_d
        return Wd

    # =====================================
    # Collision Step
    # =====================================
    @ti.kernel
    def collide(self):
        """Perform collision with solid-fluid coupling based on local weight."""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue
            self.compute_feq(i, j, k)
            self.computeomega(i, j, k)
            if self.volfrac[i, j, k] > 0.0:
                self.collide_solid(i, j, k)
            else:
                self.collide_fluid(i, j, k)

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
        for q in ti.static(range(self.Q)):
            q_inv = ti.static(self.QINV_STATIC[q])
            Omega_s = (
                self.f[i, j, k][q_inv]
                - self.feq[i, j, k][q_inv]
                + self.feqsolid[i, j, k][q]
                - self.f[i, j, k][q]
            )
            Omega_f = -self.omega[i, j, k] * (self.f[i, j, k][q] - self.feq[i, j, k][q])
            self.fpc[i, j, k][q] = (
                self.f[i, j, k][q]
                + self.weight[i, j, k] * Omega_s
                + (1.0 - self.weight[i, j, k]) * Omega_f
            )

    @ti.func
    def compute_feq_solid(self, i: int, j: int, k: int):
        """Compute equilibrium distribution using solid velocity."""
        u = self.velsolid[i, j, k]
        uv = tm.dot(u, u)
        for q in ti.static(range(self.Q)):
            cu = tm.dot(self.c[q], u)
            self.feqsolid[i, j, k][q] = self.w[q] * self.rho[i, j, k] * (
                1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uv
            )

    # =====================================
    # Interpolate Forces to Particles
    # =====================================

    @ti.kernel
    def lattice2grains(self):
        """Interpolate hydrodynamic drag force from lattice to DEM particles."""
        self.dem.gf.force_fluid.fill(0.0)

        for id in ti.ndrange(self.dem.gf.shape[0]):
            xc = (self.dem.gf[id].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[id].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[id].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx

            x_begin = ti.max(0, int(xc - 2))
            x_end = ti.min(self.Nx, int(xc + 2))
            y_begin = ti.max(0, int(yc - 2))
            y_end = ti.min(self.Ny, int(yc + 2))
            z_begin = ti.max(0, int(zc - 2))
            z_end = ti.min(self.Nz, int(zc + 2))

            fluid_vel_particle = Vector3(0.0, 0.0, 0.0)
            volfrac_particle = 0.0

            for i in range(x_begin, x_end):
                for j in range(y_begin, y_end):
                    for k in range(z_begin, z_end):
                        if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                            continue
                        dist = ti.sqrt((xc - i) ** 2 + (yc - j) ** 2 + (zc - k) ** 2)
                        weight = self.threedelta(dist)
                        fluid_vel_particle += self.vel[i, j, k] * weight
                        volfrac_particle += self.volfrac[i, j, k] * weight

            if volfrac_particle > 1e-10:
                eps_p = volfrac_particle
                d_p = 2.0 * self.dem.gf[id].radius
                fluid_vel = fluid_vel_particle * self.unit.dx / self.unit.dt
                u_slip = self.dem.gf[id].velocity - fluid_vel
                F_d = self.compute_drag_force(d_p, u_slip, eps_p)
                self.dem.gf[id].force_fluid += F_d

    # =====================================
    # Drag Force Model
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

        Reference: Tenneti et al., Int. J. Multiphase Flow 37 (2011) 1072–1092.
        """
        u_slip_mag = tm.length(u_slip)
        rho_f = self.rho0
        mu0 = self.mu
        Re_p = (1.0 - svf) * rho_f * dp * u_slip_mag / mu0

        C_d = 0.0
        if 1.0 - svf > 1e-9:
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

    # ===========================================#
    # ----- Compute Strain Rate Magnitude ----- #
    # ===========================================#
    @ti.func
    def compute_strain_rate(self, i: int, j: int, k: int):
        """Calculates the local strain-rate according to the momentum flux tensor.

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """
        # Calculate non-equilibrium momentum flux tensor components
        # Diagonal elements
        PiNeq_xx = 0.0
        PiNeq_yy = 0.0
        PiNeq_zz = 0.0

        # Off-diagonal elements
        PiNeq_xy = 0.0
        PiNeq_xz = 0.0
        PiNeq_yz = 0.0

        for q in ti.static(range(self.Q)):
            fnoneq = self.f[i, j, k][q] - self.feq[i, j, k][q]
            cx = BasicLattice3D.c[q].x
            cy = BasicLattice3D.c[q].y
            cz = BasicLattice3D.c[q].z

            PiNeq_xx += fnoneq * cx * cx
            PiNeq_yy += fnoneq * cy * cy
            PiNeq_zz += fnoneq * cz * cz
            PiNeq_xy += fnoneq * cx * cy
            PiNeq_xz += fnoneq * cx * cz
            PiNeq_yz += fnoneq * cy * cz

        # calculate the magnitude of momentum flux tensor
        Pi = (2.0 * (PiNeq_xx ** 2 + PiNeq_yy ** 2 + PiNeq_zz ** 2 + 2.0 * (
                PiNeq_xy ** 2 + PiNeq_xz ** 2 + PiNeq_yz ** 2))) ** 0.5

        # assign the local strain-rate
        self.rate[i, j, k] = self.omega[i, j, k] / (2.0 * BasicLattice3D.cssq * self.rho[i, j, k]) * Pi

    @ti.func
    def computeomega(self, i: int, j: int, k: int):
        """Compute local relaxation frequency based on effective viscosity
        """

        # Compute strain rate magnitude using lattice-specific method
        self.compute_strain_rate(i, j, k)
        # Compute turbulent viscosity: nu_t = (Cs * Delta)^2 * |S|
        nuTurb = (self.SmagorinskyConstant * 1.0) ** 2 * self.rate[i, j, k]

        # Effective relaxation time: tau_eff = tau0 + nu_turb / cs^2
        tauEff = 1.0 / self.omega0 + nuTurb / BasicLattice3D.cssq
        self.omega[i, j, k] = 1 / tauEff

    # =====================================
    # Interpolation Kernel Function
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

    # =====================================
    # High-Level Interface
    # =====================================

    def initialize_complete(self):
        """Complete initialization sequence for coupling."""
        self.initialize()
        self.grains2lattice()
        self.compute_weight()

    def update_coupling(self):
        """Update coupling fields at each time step."""
        self.grains2lattice()
        self.compute_weight()
        self.lattice2grains()