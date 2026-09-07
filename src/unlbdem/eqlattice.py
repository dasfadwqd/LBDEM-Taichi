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
    DRAG_TENNETI = 0
    DRAG_WEN_YU = 1
    DRAG_GIDASPOW = 2
    DRAG_EMMS = 3
    DRAG_BEETSTRA = 4
    ACTIVE_DRAG_MODEL = 0

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
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP| CellType.VEL_INLET_LADD):
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
            # All directions clamped (walls). Mirror-particle treatment handles
            # boundary correction via _kernel_with_mirror.
            i_min = ti.max(0, ti.cast(xc - support, ti.i32))
            i_max = ti.min(self.Nx, ti.cast(xc + support + 1, ti.i32))
            j_min = ti.max(0, ti.cast(yc - support, ti.i32))
            j_max = ti.min(self.Ny, ti.cast(yc + support + 1, ti.i32))
            k_min = ti.max(0, ti.cast(zc - support, ti.i32))
            k_max = ti.min(self.Nz, ti.cast(zc + support + 1, ti.i32))

            # # ---- Original periodic Y/Z ----
            # j_lo = ti.cast(yc - support, ti.i32)
            # j_hi = ti.cast(yc + support + 1, ti.i32)
            # k_lo = ti.cast(zc - support, ti.i32)
            # k_hi = ti.cast(zc + support + 1, ti.i32)

            # ── Pass 1: denominator sum_j W_bar_{i,j} * V_j  (Eq.16) ──
            denom = 0.0
            for ii in range(i_min, i_max):
                for jj in range(j_min, j_max):
                    for kk in range(k_min, k_max):
                        if self.CT[ii, jj, kk] & (CellType.OBSTACLE
                                                  | CellType.VEL_LADD
                                                  | CellType.FREE_SLIP
                        | CellType.VEL_INLET_LADD):
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
                                                  | CellType.FREE_SLIP
                        | CellType.VEL_INLET_LADD):
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
        Compute the dimensionless drag weight used in the collision operator.

        The selected drag model provides Cd_star in:
            F_drag = -3*pi*mu*dp*eps_f*Cd_star*u_slip
        In lattice units this gives:
            F_lattice = -Wd*u_slip_lattice
            Wd = 3*pi*dp_lattice*nu_lattice*eps_f*Cd_star
        """
        u_slip_mag = tm.length(u_slip)
        Re_p = self._particle_reynolds(dp, u_slip_mag, svf)
        fvf = 1.0 - svf
        Cd_star = self._drag_coefficient_star(Re_p, svf)
        dp_lattice = dp / self.unit.dx
        Wd = 3.0 * tm.pi * dp_lattice * self.nuLu * fvf * Cd_star
        return Wd

    @ti.func
    def _particle_reynolds(self, dp: float, u_slip_mag: float, svf: float) -> float:
        return (1.0 - svf) * self.rho0 * dp * u_slip_mag / self.mu

    @ti.func
    def _drag_coefficient_star(self, Re_p: float, svf: float) -> float:
        """
        Return Cd_star in F_drag = -3*pi*mu*dp*eps_f*Cd_star*u_slip.

        This keeps physical drag and lattice weight exactly consistent.
        Wen-Yu uses:
            Cd = 24/Re_p*(1 + 0.15*Re_p^0.687)
            |F| = 0.5*Cd*eps_f^-2.7*pi*R^2*rho_f*|u|^2
        which gives Cd_star = (1 + 0.15*Re_p^0.687)*eps_f^-4.7
        under the Re_p definition used here.
        Gidaspow uses Wen-Yu for eps_f > 0.8 and Ergun otherwise.
        EMMS uses its heterogeneity correction for eps_f > 0.74 and Ergun
        otherwise; its high-voidage branch gives Cd_star = Re_p*omega/(24*eps_f).
        Beetstra directly returns Cd_star using svf as the solid volume fraction.
        """
        Cd_star = 0.0
        fvf = 1.0 - svf
        if ti.static(EqIMBlattice3D.ACTIVE_DRAG_MODEL == EqIMBlattice3D.DRAG_WEN_YU):
            if fvf > 1e-9:
                Cd_star = (1.0 + 0.15 * tm.pow(Re_p, 0.687)) / tm.pow(fvf, 4.7)
        elif ti.static(EqIMBlattice3D.ACTIVE_DRAG_MODEL == EqIMBlattice3D.DRAG_GIDASPOW):
            if fvf > 1e-9:
                if fvf > 0.8:
                    Cd_star = (1.0 + 0.15 * tm.pow(Re_p, 0.687)) / tm.pow(fvf, 3.65)
                else:
                    Cd_star = self._ergun_coefficient_star(Re_p, svf)
        elif ti.static(EqIMBlattice3D.ACTIVE_DRAG_MODEL == EqIMBlattice3D.DRAG_EMMS):
            if fvf > 1e-9:
                if fvf > 0.74:
                    emms_factor = 0.0
                    if fvf <= 0.82:
                        emms_factor = 0.0214 / (4.0 * (fvf - 0.7463) ** 2 + 0.0044) - 0.5760
                    elif fvf <= 0.97:
                        emms_factor = 0.0038 / (4.0 * (fvf - 0.7789) ** 2 + 0.0040) - 0.0101
                    else:
                        emms_factor = 32.8295 * fvf - 31.8295
                    Cd_star = Re_p * emms_factor / (24.0 * fvf)
                else:
                    Cd_star = self._ergun_coefficient_star(Re_p, svf)
        elif ti.static(EqIMBlattice3D.ACTIVE_DRAG_MODEL == EqIMBlattice3D.DRAG_BEETSTRA):
            if fvf > 1e-9:
                Re_eff = ti.max(Re_p, 1e-12)
                static_term = 10.0 * svf / (fvf ** 2) + fvf ** 2 * (1.0 + 1.5 * ti.sqrt(svf))
                dynamic_num = (1.0 / fvf
                               + 3.0 * svf * fvf
                               + 8.4 * tm.pow(Re_eff, -0.343))
                dynamic_den = (1.0
                               + tm.pow(10.0, 3.0 * svf)
                               * tm.pow(Re_eff, -(1.0 + 4.0 * svf) / 2.0))
                dynamic_term = (
                    0.413 * Re_eff / (24.0 * fvf ** 2)
                    * dynamic_num / dynamic_den
                )
                Cd_star = static_term + dynamic_term
        else:
            if fvf > 1e-9:
                Cd0 = 1.0 + 0.15 * tm.pow(Re_p, 0.687)
                A_eps = (5.81 * svf / (fvf ** 3)
                         + 0.48 * tm.pow(svf, 1.0 / 3.0) / (fvf ** 4))
                svf3 = svf ** 3
                B_eps = svf3 * Re_p * (0.95 + 0.61 * svf3 / (fvf ** 2))
                Cd_star = fvf * (Cd0 / (fvf ** 3) + A_eps + B_eps)
        return Cd_star

    @ti.func
    def _ergun_coefficient_star(self, Re_p: float, svf: float) -> float:
        fvf = 1.0 - svf
        Cd_star = 0.0
        if fvf > 1e-9:
            Cd_star = (25.0 / 3.0) * svf / (fvf ** 2) + (1.75 / 18.0) * Re_p / (fvf ** 2)
        return Cd_star

    # =====================================
    # Collision Step
    # =====================================
    @ti.kernel
    def collide(self):
        """Perform collision with solid-fluid coupling based on local weight."""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                continue
            #self.computeOmega(i , j, k)
            self.compute_feq(i, j, k)
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
        Solid-fluid collision operator with mirror-bounce-back coupling.

        Combines fluid relaxation (Ω_f) and solid momentum exchange (Ω_s).

        Args:
            i, j, k (int): Lattice indices.
        """
        # Update equilibrium using solid velocity
        self.compute_feq_solid(i, j, k)
        for q in range(EqIMBlattice3D.Q):

            Omega_s = (
                self.f[i, j, k][EqIMBlattice3D.qinv[q]]
                - self.feq[i, j, k][EqIMBlattice3D.qinv[q]]
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

            # Support region (threedelta: +/- 1.5 lu)
            # All directions clamped (walls).
            x_begin = ti.max(0, ti.cast(xc - support, ti.i32))
            x_end = ti.min(self.Nx, ti.cast(xc + support + 1, ti.i32))
            y_begin = ti.max(0, ti.cast(yc - support, ti.i32))
            y_end = ti.min(self.Ny, ti.cast(yc + support + 1, ti.i32))
            z_begin = ti.max(0, ti.cast(zc - support, ti.i32))
            z_end = ti.min(self.Nz, ti.cast(zc + support + 1, ti.i32))

            # # ---- Original periodic Y/Z ----
            # y_lo = ti.cast(yc - support, ti.i32)
            # y_hi = ti.cast(yc + support + 1, ti.i32)
            # z_lo = ti.cast(zc - support, ti.i32)
            # z_hi = ti.cast(zc + support + 1, ti.i32)

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
                                                  | CellType.FREE_SLIP
                        | CellType.VEL_INLET_LADD):
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
        Compute drag force on a particle using the selected drag model.

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
        Re_p = self._particle_reynolds(dp, u_slip_mag, svf)
        fvf = 1.0 - svf
        Cd_star = self._drag_coefficient_star(Re_p, svf)

        F_drag = -3.0 * tm.pi * dp * self.mu * fvf * Cd_star * u_slip

        return F_drag



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

        All walls (X, Y, Z) use the mirror-particle treatment (Eq.22 of Zhu et al. 2026):
          W_bar = W(x_p) + W(x'_p)
        where x'_p is the mirror image of x_p reflected about the nearest wall.
        This folds the truncated kernel lobe back into the domain, preventing
        underestimation of solid volume fraction at boundary nodes.

        Args:
            xc, yc, zc (float): Particle centre in lattice coordinates.
            ii, jj, kk (int):   Target lattice node (all clamped to [0, Ni)).

        Returns:
            float: Kernel weight W_bar (primary + mirror corrections).
        """
        support = 1.5  # threedelta support radius in lattice units
        dist = ti.sqrt((xc - ii) ** 2 + (yc - jj) ** 2 + (zc - kk) ** 2)

        # W(x_p) -- primary contribution
        w_primary = self.threedelta(dist)

        # W(x'_p) -- mirror contributions (all walls)
        # Walls sit at HALF-cell planes: DEM domain maps to lattice
        # coords [0.5, Ni-1.5], so mirror about 0.5 / Ni-1.5.
        w_mirror = 0.0
        # X walls
        if xc < 0.5 + support:  # near left wall (xc = 0.5)
            xc_mir = 1.0 - xc
            dist = ti.sqrt((xc_mir - ii) ** 2 + (yc - jj) ** 2 + (zc - kk) ** 2)
            w_mirror += self.threedelta(dist)
        if xc > float(self.Nx) - 1.5 - support:  # near right wall (xc = Nx-1.5)
            xc_mir = 2.0 * (float(self.Nx) - 1.5) - xc
            dist = ti.sqrt((xc_mir - ii) ** 2 + (yc - jj) ** 2 + (zc - kk) ** 2)
            w_mirror += self.threedelta(dist)
        # Y walls
        if yc < 0.5 + support:  # near bottom wall (yc = 0.5)
            yc_mir = 1.0 - yc
            dist = ti.sqrt((xc - ii) ** 2 + (yc_mir - jj) ** 2 + (zc - kk) ** 2)
            w_mirror += self.threedelta(dist)
        if yc > float(self.Ny) - 1.5 - support:  # near top wall (yc = Ny-1.5)
            yc_mir = 2.0 * (float(self.Ny) - 1.5) - yc
            dist = ti.sqrt((xc - ii) ** 2 + (yc_mir - jj) ** 2 + (zc - kk) ** 2)
            w_mirror += self.threedelta(dist)
        # Z walls
        if zc < 0.5 + support:  # near back wall (zc = 0.5)
            zc_mir = 1.0 - zc
            dist = ti.sqrt((xc - ii) ** 2 + (yc - jj) ** 2 + (zc_mir - kk) ** 2)
            w_mirror += self.threedelta(dist)
        if zc > float(self.Nz) - 1.5 - support:  # near front wall (zc = Nz-1.5)
            zc_mir = 2.0 * (float(self.Nz) - 1.5) - zc
            dist = ti.sqrt((xc - ii) ** 2 + (yc - jj) ** 2 + (zc_mir - kk) ** 2)
            w_mirror += self.threedelta(dist)

        return w_primary + w_mirror  # W_bar = W(x_p) + W(x'_p)
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
