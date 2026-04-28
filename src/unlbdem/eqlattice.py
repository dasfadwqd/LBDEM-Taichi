"""
Unresolved LBM-DEM coupling simulation based on the particle equivalence method
proposed by Prof. Limin Wang (DOI: https://doi.org/10.1016/j.cej.2023.142898).
https://doi.org/10.1016/j.ces.2026.123562
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
        Map particle data (eps_s, u_s) to lattice using threedelta kernel with
        mirror-particle boundary treatment.

        Replaces the original correction_factor approach with the normalized weight
        formulation of Zhu et al. (2026):

          Eq.(16): w_{i,j} = W_bar_{i,j} * V_j / sum_j W_bar_{i,j} * V_j
                   (V_j = 1 lu^3; W_bar includes mirror correction per Eq.22)
          Eq.(17): eps_{i,j} = w_{i,j} * V_{part,i} / V_lattice
          Eq.(18): u_{i,j}   = u_{part,i} * eps_{i,j}

        After accumulation over all particles:
          u_s[node] = sum(u_{i,j}) / sum(eps_{i,j})   (momentum-weighted solid velocity)
        """
        self.volfrac.fill(0.0)
        self.velsolid.fill(0.0)
        self.velsum.fill(0.0)
        self.weight_sum.fill(0.0)

        V_lattice = self.unit.dx ** 3
        support = 1.5  # threedelta support radius in lattice units

        for pid in range(self.dem.gf.shape[0]):
            # Physical position -> lattice coordinates
            xc = (self.dem.gf[pid].position[0] - self.dem.config.domain.xmin
                  + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[pid].position[1] - self.dem.config.domain.ymin
                  + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[pid].position[2] - self.dem.config.domain.zmin
                  + 0.5 * self.unit.dx) / self.unit.dx

            V_part = 4.0 / 3.0 * tm.pi * self.dem.gf[pid].radius ** 3
            vel_lu = self.dem.gf[pid].velocity * self.unit.dt / self.unit.dx

            # Support region (threedelta: +/- 1.5 lu)
            i_min = ti.max(0, ti.cast(xc - support, ti.i32))
            i_max = ti.min(self.Nx, ti.cast(xc + support + 1, ti.i32))
            j_min = ti.max(0, ti.cast(yc - support, ti.i32))
            j_max = ti.min(self.Ny, ti.cast(yc + support + 1, ti.i32))
            k_min = ti.max(0, ti.cast(zc - support, ti.i32))
            k_max = ti.min(self.Nz, ti.cast(zc + support + 1, ti.i32))

            # ── Pass 1: denominator sum_j W_bar_{i,j} * V_j  (Eq.16) ──
            denom = 0.0
            for ii in range(i_min, i_max):
                for jj in range(j_min, j_max):
                    for kk in range(k_min, k_max):
                        if self.CT[ii, jj, kk] & (CellType.OBSTACLE
                                                  | CellType.VEL_LADD
                                                  | CellType.FREE_SLIP):
                            continue
                        w_bar = self._kernel_with_mirror(xc, yc, zc, ii, jj, kk)
                        if w_bar < 0.0:
                            continue
                        denom += w_bar  # V_j = 1 lu^3

            if denom < 1e-30:
                continue  # no valid neighbours

            # ── Pass 2: distribute using normalized weight (Eq.16-18) ──
            for ii in range(i_min, i_max):
                for jj in range(j_min, j_max):
                    for kk in range(k_min, k_max):
                        if self.CT[ii, jj, kk] & (CellType.OBSTACLE
                                                  | CellType.VEL_LADD
                                                  | CellType.FREE_SLIP):
                            continue
                        w_bar = self._kernel_with_mirror(xc, yc, zc, ii, jj, kk)
                        if w_bar < 0.0:
                            continue

                        w_norm = w_bar / denom  # Eq.(16)
                        eps_ij = w_norm * V_part / V_lattice  # Eq.(17)
                        u_ij = vel_lu * eps_ij  # Eq.(18)

                        ti.atomic_add(self.volfrac[ii, jj, kk], eps_ij)
                        ti.atomic_add(self.velsum[ii, jj, kk], u_ij)
                        ti.atomic_add(self.weight_sum[ii, jj, kk], w_norm)

        # ── Recover solid velocity u_s = sum(u_{i,j}) / sum(eps_{i,j}) ──
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.volfrac[i, j, k] >= 1.0:
                self.volfrac[i, j, k] = 0.99
                print("Warning: volfrac[{}, {}, {}] >= 1.0".format(i, j, k))
            if self.volfrac[i, j, k] > 1e-15:
                self.velsolid[i, j, k] = self.velsum[i, j, k] / self.volfrac[i, j, k]

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
        for q in range(EqIMBlattice3D.Q):
            Omega_s = (self.f[i, j, k][EqIMBlattice3D.qinv[q]] - self.feq[i, j, k][EqIMBlattice3D.qinv[q]] +
                       self.feqsolid[i, j, k][q] - self.f[i, j, k][q])
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
        """
        Interpolate fluid properties from lattice to DEM particle positions.

        Implements Eq.(19-21) of Zhu et al. (2026):
          Eq.(19): u_fluid_i = sum_j (w_{i,j} * u_j)   / sum_j w_{i,j}
          Eq.(20): rho_fluid_i = sum_j (w_{i,j} * rho_j) / sum_j w_{i,j}
          Eq.(21): eps_fluid_i = sum_j eps_{i,j} / N_lattice  (arithmetic mean)

        The same mirror-extended kernel W_bar (Eq.22) is used for consistency
        with grains2lattice (fluid->solid range = solid->fluid range, Sec.3.1.3).
        """
        self.dem.gf.force_fluid.fill(0.0)

        support = 1.5  # threedelta support radius in lattice units

        for pid in ti.ndrange(self.dem.gf.shape[0]):
            xc = (self.dem.gf[pid].position[0] - self.dem.config.domain.xmin
                  + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[pid].position[1] - self.dem.config.domain.ymin
                  + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[pid].position[2] - self.dem.config.domain.zmin
                  + 0.5 * self.unit.dx) / self.unit.dx

            x_begin = ti.max(0, ti.cast(xc - support, ti.i32))
            x_end = ti.min(self.Nx, ti.cast(xc + support + 1, ti.i32))
            y_begin = ti.max(0, ti.cast(yc - support, ti.i32))
            y_end = ti.min(self.Ny, ti.cast(yc + support + 1, ti.i32))
            z_begin = ti.max(0, ti.cast(zc - support, ti.i32))
            z_end = ti.min(self.Nz, ti.cast(zc + support + 1, ti.i32))

            # Accumulators
            vel_wsum = Vector3(0.0, 0.0, 0.0)  # sum w_{i,j}*u_j   — Eq.(19) numerator
            rho_wsum = 0.0  # sum w_{i,j}*rho_j — Eq.(20) numerator
            eps_sum = 0.0  # sum eps_{i,j}     — Eq.(21) numerator
            w_total = 0.0  # sum w_{i,j}       — Eq.(19,20) denominator
            n_lattice = 0  # N_lattice         — Eq.(21) denominator

            for ii in range(x_begin, x_end):
                for jj in range(y_begin, y_end):
                    for kk in range(z_begin, z_end):
                        if self.CT[ii, jj, kk] & (CellType.OBSTACLE
                                                  | CellType.VEL_LADD
                                                  | CellType.FREE_SLIP):
                            continue
                        w_ij = self._kernel_with_mirror(xc, yc, zc, ii, jj, kk)
                        if w_ij < 0.0:
                            continue

                        vel_wsum += self.vel[ii, jj, kk] * w_ij
                        rho_wsum += self.rho[ii, jj, kk] * w_ij
                        eps_sum +=(1.0 - self.volfrac[ii, jj, kk] )
                        w_total += w_ij
                        n_lattice += 1

            # Recover fluid quantities at particle location
            fluid_vel = Vector3(0.0, 0.0, 0.0)
            eps_fluid = 0.0

            if w_total > 1e-15:
                # Eq.(19): u_fluid = sum(w*u) / sum(w), convert lu -> physical
                fluid_vel = vel_wsum / w_total * self.unit.dx / self.unit.dt
                # Eq.(20): rho_fluid = sum(w*rho) / sum(w)  [available if needed]
                # fluid_rho = rho_wsum / w_total

            if n_lattice > 0:
                # Eq.(21): eps_fluid = sum(eps_{i,j}) / N_lattice
                eps_fluid = eps_sum / float(n_lattice)

            # Drag force
            if 1.0 - eps_fluid > 1e-15:
                eps_p = 1.0 - eps_fluid
                d_p = 2.0 * self.dem.gf[pid].radius
                u_slip = self.dem.gf[pid].velocity - fluid_vel
                F_d = self.compute_drag_force(d_p, u_slip, eps_p)
                self.dem.gf[pid].force_fluid += F_d

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
    # Mirror-Particle Kernel Helper
    # =====================================

    @ti.func
    def _kernel_with_mirror(self, xc: float, yc: float, zc: float,
                            ii: int, jj: int, kk: int) -> float:
        """
        Evaluate mirror-extended kernel W_bar = W(x_p) + W(x'_p) at lattice node (ii,jj,kk).

        Implements boundary treatment Eq.(22) of Zhu et al. (2026):
          W_bar = W(x_p) + W(x'_p)
        where x'_p is the mirror image of x_p reflected about the nearest domain wall.

        This folds the truncated kernel lobe back into the domain, preventing
        underestimation of solid volume fraction at boundary nodes (cf. Fig.4 in paper).
        The threedelta kernel has support radius 1.5 lu, so mirror correction is
        triggered when the particle centre is within 1.5 lu of any wall.

        Args:
            xc, yc, zc (float): Particle centre in lattice coordinates.
            ii, jj, kk (int):   Target lattice node indices.

        Returns:
            float: Mirror-corrected kernel weight W_bar.
        """
        support = 1.5  # threedelta support radius in lattice units

        # Primary distances
        dx_p = ti.abs(xc - ii)
        dy_p = ti.abs(yc - jj)
        dz_p = ti.abs(zc - kk)

        dist = ti.sqrt((xc - ii) ** 2 + (yc - jj) ** 2 + (zc - kk) ** 2)

        # W(x_p) -- primary contribution
        w_primary = self.threedelta(dist)

        # W(x'_p) -- mirror contributions (Eq.22)
        w_mirror = 0.0

        # -- x walls --
        if xc < support:  # near left wall (i = 0)
            xc_mir = -xc
            dist = ti.sqrt((xc_mir - ii) ** 2 + (yc - jj) ** 2 + (zc - kk) ** 2)
            w_mirror += self.threedelta(dist)
        if xc > float(self.Nx) - support:  # near right wall (i = Nx)
            xc_mir = 2.0 * float(self.Nx) - xc
            dist = ti.sqrt((xc_mir - ii) ** 2 + (yc - jj) ** 2 + (zc - kk) ** 2)
            w_mirror += self.threedelta(dist)

        # -- y walls --
        if yc < support:
            yc_mir = -yc
            dist = ti.sqrt((yc_mir - ii) ** 2 + (yc - jj) ** 2 + (zc - kk) ** 2)
            w_mirror += self.threedelta(dist)
        if yc > float(self.Ny) - support:
            yc_mir = 2.0 * float(self.Ny) - yc
            dist = ti.sqrt((yc_mir - ii) ** 2 + (yc - jj) ** 2 + (zc - kk) ** 2)
            w_mirror += self.threedelta(dist)

        # -- z walls --
        if zc < support:
            zc_mir = -zc
            dist = ti.sqrt((zc_mir - ii) ** 2 + (yc - jj) ** 2 + (zc - kk) ** 2)
            w_mirror += self.threedelta(dist)
        if zc > float(self.Nz) - support:
            zc_mir = 2.0 * float(self.Nz) - zc
            dist = ti.sqrt((zc_mir - ii) ** 2 + (yc - jj) ** 2 + (zc - kk) ** 2)
            w_mirror += self.threedelta(dist)

        return w_primary + w_mirror  # W_bar = W(x_p) + W(x'_p),
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