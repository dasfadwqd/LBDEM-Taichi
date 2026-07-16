"""
Hybrid Resolved/Unresolved LBM-DEM Coupling (3D)
=================================================

This module implements a hybrid Fluid-Structure Interaction (FSI) solver combining:
1. Partially Saturated Cell (PSC) method for coarse/resolved particles (dp > dx).
2. Particle Equivalence IBM (Tenneti drag) for fine/unresolved particles (dp ≤ dx).

Key Design Decisions
--------------------
* Volume Fraction Splitting:
    - `coarse_volfrac`: Contribution from resolved particles (PSC).
    - `fine_volfrac`:   Contribution from unresolved particles (IBM).
* Collision Dispatch (per-cell priority):
    - Pure Fluid:     Standard BGK.
    - Coarse only:    PSC bounce-back collision.
    - Fine only:      IBM weighted collision (Tenneti drag model).
    - Hybrid (both):  Weighted combination of PSC and IBM operators.
* Force Transfer:
    - Unified `lattice2grains()` handles both particle types.
    - Coarse: PSC momentum exchange accumulated from collision.
    - Fine:   Tenneti drag force via IBM interpolation of fluid velocity.
* Kernel:
    - Fine particles use Peskin 3-point delta kernel (radial form) with
      mirror-particle boundary treatment (Zhu et al. 2026, Eq.22).

References
----------
* PSC Method:     Noble & Torczynski, Int. J. Mod. Phys. C 9 (1998) 1189-1201
* Tenneti Drag:   Tenneti et al., Int. J. Multiphase Flow 37 (2011) 1072-1092
* Equiv. IBM:     Wang et al., Chem. Eng. J. (2023) 142898
* Mirror Kernel:  Zhu et al., Chem. Eng. Sci. (2026) 123562
"""

import taichi as ti
import taichi.math as tm

from src.lbm3d.lbm_solver3d import BasicLattice3D
from src.lbm3d.lbmutils import CellType
from src.dem3d.demsolver import DEMSolver

# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------
Vector3 = ti.types.vector(3, float)


class HybridLattice3D(BasicLattice3D):
    """
    3-D LBM lattice with hybrid resolved/unresolved FSI coupling.

    Particle Classification (Runtime)
    ---------------------------------
    For each DEM grain:
        - Coarse (Resolved):   2 * radius >  dx  → PSC Method (psclattice3d)
        - Fine (Unresolved):   2 * radius ≤ dx  → IBM/Tenneti Method (eqlattice)

    This enables polydisperse simulations where each grain is handled by the
    most appropriate coupling strategy automatically.
    """

    def __init__(
        self,
        Nx: int, Ny: int, Nz: int,
        omega: float,
        dx: float, dt: float,
        rho: float,
        dem_solver: DEMSolver,
    ):
        super().__init__(Nx, Ny, Nz, omega, dx, dt, rho)
        shape = (Nx, Ny, Nz)

        # =====================================================================
        # 1. Fluid Properties (Physical & Lattice Units)
        # =====================================================================
        self.rho0 = rho
        self.omega0 = omega
        # Lattice kinematic viscosity: nu_L = (1/omega - 0.5) * c_s^2  (c_s^2 = 1/3)
        self.nuLu = (1.0 / omega - 0.5) / 3.0
        # Physical kinematic viscosity: nu_P = nu_L * dx^2 / dt
        self.nu = self.nuLu * (dx ** 2) / dt
        # Dynamic viscosity: mu = rho * nu_P
        self.mu = rho * self.nu

        # =====================================================================
        # 2. Coupling Fields — Coarse (PSC, resolved)
        # =====================================================================
        self.coarse_id = ti.field(int, shape=shape)           # grain ID map (-1 = none)
        self.coarse_volfrac = ti.field(float, shape=shape)    # solid volume fraction ε_c
        self.coarse_weight = ti.field(float, shape=shape)     # PSC weight B
        self.coarse_velsolid = ti.Vector.field(self.D, float, shape=shape)
        self.coarse_feqsolid = ti.Vector.field(self.Q, float, shape=shape)

        # =====================================================================
        # 3. Coupling Fields — Fine (IBM, unresolved)
        # =====================================================================
        self.fine_volfrac = ti.field(float, shape=shape)      # solid volume fraction ε_f
        self.fine_weight = ti.field(float, shape=shape)       # IBM weight β
        self.fine_velsolid = ti.Vector.field(self.D, float, shape=shape)
        self.fine_feqsolid = ti.Vector.field(self.Q, float, shape=shape)

        # Temporary accumulators for fine-grain mapping
        self._fine_velsum = ti.Vector.field(self.D, float, shape=shape)
        self._fine_weightsum = ti.field(float, shape=shape)

        # =====================================================================
        # 4. Force & Momentum Exchange Fields
        # =====================================================================
        self.hydroforce = ti.Vector.field(self.D, float, shape=shape)
        self.hydrotorque = ti.Vector.field(self.D, float, shape=shape)

        # =====================================================================
        # 5. DEM Solver Reference
        # =====================================================================
        self.dem = dem_solver

    # =====================================================================
    # Initialization
    # =====================================================================

    @ti.kernel
    def initialize(self):
        """Initialize distribution functions to equilibrium and map grains."""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD
                                   | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                continue
            self.compute_feq(i, j, k)
            for q in ti.static(range(HybridLattice3D.Q)):
                self.f[i, j, k][q] = self.feq[i, j, k][q]

    # =====================================================================
    # DEM → Lattice Mapping  —  Coarse (PSC)
    # =====================================================================

    @ti.kernel
    def map_coarse_grains(self):
        """
        Map resolved (coarse) grains using the PSC solid-fraction method.

        Uses a 5×5×5 sub-cell decomposition to estimate solid volume fraction ε
        for partially covered cells. Fully covered cells get ε = 1.
        Only grains with diameter > dx contribute here.

        Also computes PSC weight B and solid velocity at each covered node.
        """
        # Reset fields
        self.coarse_id.fill(-1)
        self.coarse_volfrac.fill(0.0)
        self.coarse_weight.fill(0.0)
        self.coarse_velsolid.fill(0.0)

        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD
                                   | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                continue

            for gid in range(self.dem.gf.shape[0]):
                # Skip fine particles
                if 2.0 * self.dem.gf[gid].radius <= self.unit.dx:
                    continue

                # Grain center in lattice coordinates
                xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin
                      + 0.5 * self.unit.dx) / self.unit.dx
                yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin
                      + 0.5 * self.unit.dx) / self.unit.dx
                zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin
                      + 0.5 * self.unit.dx) / self.unit.dx

                # Effective radius (shrunk by 0.5dx for boundary-cell treatment)
                r = (self.dem.gf[gid].radius - 0.5 * self.unit.dx) / self.unit.dx
                dist = ti.sqrt((xc - i) ** 2 + (yc - j) ** 2 + (zc - k) ** 2)

                # Case 1: Fully outside
                if dist >= r + 0.5 * ti.sqrt(3.0):
                    continue

                # Case 2: Fully inside
                if dist <= r - 0.5 * ti.sqrt(3.0):
                    self.coarse_id[i, j, k] = gid
                    self.coarse_volfrac[i, j, k] = 1.0
                    self._psc_set_weight(i, j, k)
                    self._psc_set_velsolid(i, j, k, gid, xc, yc, zc)
                    break  # One grain dominates per cell

                # Case 3: Partially covered — 5³ sub-cell integration
                cnt = 0
                for si in range(5):
                    for sj in range(5):
                        for sk in range(5):
                            sub_x = i - 0.4 + 0.2 * si
                            sub_y = j - 0.4 + 0.2 * sj
                            sub_z = k - 0.4 + 0.2 * sk
                            d2 = ti.sqrt((xc - sub_x) ** 2 + (yc - sub_y) ** 2 + (zc - sub_z) ** 2)
                            if d2 < r:
                                cnt += 1

                eps = cnt / 125.0

                # Keep the grain with the largest coverage per cell
                if eps > self.coarse_volfrac[i, j, k]:
                    self.coarse_id[i, j, k] = gid
                    self.coarse_volfrac[i, j, k] = eps
                    self._psc_set_weight(i, j, k)
                    self._psc_set_velsolid(i, j, k, gid, xc, yc, zc)

    @ti.func
    def _psc_set_weight(self, i: int, j: int, k: int):
        """
        PSC weighting coefficient B.
        B = ε·(τ-½) / [(1-ε) + (τ-½)]
        """
        eps = self.coarse_volfrac[i, j, k]
        tau_m = 1.0 / self.omega[i, j, k] - 0.5
        self.coarse_weight[i, j, k] = (eps * tau_m) / ((1.0 - eps) + tau_m)

    @ti.func
    def _psc_set_velsolid(self, i: int, j: int, k: int,
                          gid: int, xc: float, yc: float, zc: float):
        """
        Solid velocity at node (i,j,k) for a coarse grain.
        u_s = (u_trans + ω × r) · dt/dx   [lattice units]
        """
        r_vec = Vector3(i, j, k) - Vector3(xc, yc, zc)
        omega_cross_r = tm.cross(self.dem.gf[gid].omega, r_vec * self.unit.dx)
        self.coarse_velsolid[i, j, k] = (
            (self.dem.gf[gid].velocity + omega_cross_r) * self.unit.dt / self.unit.dx
        )

    # =====================================================================
    # DEM → Lattice Mapping  —  Fine (IBM)
    # =====================================================================

    @ti.kernel
    def map_fine_grains(self):
        """
        Map unresolved (fine) grains using Peskin kernel with mirror correction.

        Three-pass algorithm per grain:
          Pass 1: denom = Σ W_bar                 (kernel normalisation)
          Pass 2: D_corr = Σ w_norm · (1 - ε_c)   (coarse-occupation correction)
          Pass 3: ε_f = w_norm · (1-ε_c) / D_corr · V_grain / V_lattice

        Nodes fully occupied by coarse particles (ε_c ≥ 1.0) are skipped.
        Mirror-particle boundary treatment (Eq.22, Zhu et al. 2026) folds
        truncated kernel lobes back into the domain.

        Reference: eqlattice.py grains2lattice + coarse-correction extension
        """
        # Reset fields
        self.fine_volfrac.fill(0.0)
        self._fine_velsum.fill(0.0)
        self._fine_weightsum.fill(0.0)

        V_lattice = self.unit.dx ** 3
        support = 1.5

        for gid in range(self.dem.gf.shape[0]):
            # Skip coarse particles
            if 2.0 * self.dem.gf[gid].radius > self.unit.dx:
                continue

            # Grain center in lattice coordinates
            xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin
                  + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin
                  + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin
                  + 0.5 * self.unit.dx) / self.unit.dx

            V_grain = 4.0 / 3.0 * tm.pi * self.dem.gf[gid].radius ** 3
            vel_lattice = self.dem.gf[gid].velocity * self.unit.dt / self.unit.dx

            # Search bounds
            i_min = ti.max(0, ti.cast(xc - support, ti.i32))
            i_max = ti.min(self.Nx, ti.cast(xc + support + 1, ti.i32))
            j_min = ti.max(0, ti.cast(yc - support, ti.i32))
            j_max = ti.min(self.Ny, ti.cast(yc + support + 1, ti.i32))
            k_min = ti.max(0, ti.cast(zc - support, ti.i32))
            k_max = ti.min(self.Nz, ti.cast(zc + support + 1, ti.i32))

            # ----------------------------------------------------------------
            # Pass 1: kernel normalisation  denom = Σ W_bar
            # ----------------------------------------------------------------
            denom = 0.0
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    for k in range(k_min, k_max):
                        if self.CT[i, j, k] & (CellType.OBSTACLE
                                               | CellType.VEL_LADD
                                               | CellType.FREE_SLIP
                                               | CellType.VEL_INLET_LADD):
                            continue
                        if self.coarse_volfrac[i, j, k] >= 1.0:
                            continue
                        w_bar = self._kernel_with_mirror(xc, yc, zc, i, j, k)
                        if w_bar > 0.0:
                            denom += w_bar

            if denom < 1e-30:
                continue

            # ----------------------------------------------------------------
            # Pass 2: corrected denominator  D_corr = Σ w_norm · (1-ε_c)
            # ----------------------------------------------------------------
            denom_corrected = 0.0
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    for k in range(k_min, k_max):
                        if self.CT[i, j, k] & (CellType.OBSTACLE
                                               | CellType.VEL_LADD
                                               | CellType.FREE_SLIP
                                               | CellType.VEL_INLET_LADD):
                            continue
                        if self.coarse_volfrac[i, j, k] >= 1.0:
                            continue
                        w_bar = self._kernel_with_mirror(xc, yc, zc, i, j, k)
                        if w_bar <= 0.0:
                            continue
                        w_norm = w_bar / denom
                        avail = 1.0 - self.coarse_volfrac[i, j, k]
                        denom_corrected += w_norm * avail

            if denom_corrected < 1e-30:
                continue

            # ----------------------------------------------------------------
            # Pass 3: distribute corrected volume fraction and velocity
            #   ε_f = w_norm · (1-ε_c) / D_corr · V_grain / V_lattice
            # ----------------------------------------------------------------
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    for k in range(k_min, k_max):
                        if self.CT[i, j, k] & (CellType.OBSTACLE
                                               | CellType.VEL_LADD
                                               | CellType.FREE_SLIP
                                               | CellType.VEL_INLET_LADD):
                            continue
                        if self.coarse_volfrac[i, j, k] >= 1.0:
                            continue
                        w_bar = self._kernel_with_mirror(xc, yc, zc, i, j, k)
                        if w_bar <= 0.0:
                            continue

                        w_norm = w_bar / denom
                        avail = 1.0 - self.coarse_volfrac[i, j, k]
                        eps_ij = w_norm * avail / denom_corrected * V_grain / V_lattice

                        ti.atomic_add(self.fine_volfrac[i, j, k], eps_ij)
                        ti.atomic_add(self._fine_velsum[i, j, k], vel_lattice * eps_ij)
                        ti.atomic_add(self._fine_weightsum[i, j, k], w_norm)

        # ------------------------------------------------------------------
        # Normalise solid velocity; clamp volume fraction
        # ------------------------------------------------------------------
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            avail = 1.0 - self.coarse_volfrac[i, j, k]

            if self.fine_volfrac[i, j, k] > avail:
                if self.fine_volfrac[i, j, k] - avail > 1e-6:
                    print("Warning: fine volfrac overflow at ({},{},{}): "
                          "fine={}, coarse={}".format(
                              i, j, k,
                              self.fine_volfrac[i, j, k],
                              self.coarse_volfrac[i, j, k]))
                self.fine_volfrac[i, j, k] = ti.max(0.0, avail - 0.01)

            if self.fine_volfrac[i, j, k] > 1e-10:
                self.fine_velsolid[i, j, k] = (
                    self._fine_velsum[i, j, k] / self.fine_volfrac[i, j, k]
                )

    # =====================================================================
    # Weight & Drag  —  Tenneti Model (Fine particles)
    # =====================================================================

    @ti.kernel
    def compute_fine_weights(self):
        """
        Compute per-cell IBM weight β using the Tenneti drag model.

        Must be called after map_fine_grains() so fine_volfrac and
        fine_velsolid are up-to-date.

        β = W_d = 3π · d_p^L · ν_L · (1 - ε_p) · Cd(Re_p, ε_p)

        where ε_p = fine_volfrac (solid vol. frac. of fine particles),
        and the total fluid fraction is 1 - ε_c - ε_f.
        """
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD
                                   | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                continue

            svf = self.fine_volfrac[i, j, k]        # ε_p (fine solid fraction)
            fvf = 1.0 - svf - self.coarse_volfrac[i, j, k]  # total fluid fraction

            if svf > 0.0 and fvf > 1e-10:
                V_lattice = self.unit.dx ** 3
                # Effective particle radius from volume fraction
                R_eff = ti.pow(3.0 * V_lattice * svf / (4.0 * tm.pi), 1.0 / 3.0)
                d_eff = 2.0 * R_eff

                # Slip velocity in physical units
                v_slip = ((self.fine_velsolid[i, j, k] - self.vel[i, j, k])
                          * self.unit.dx / self.unit.dt)

                self.fine_weight[i, j, k] = self._weight_coefficient(d_eff, v_slip, svf)

    @ti.func
    def _weight_coefficient(self, dp: float, u_slip: Vector3, svf: float) -> float:
        """
        Compute dimensionless IBM weight coefficient from Tenneti drag model.

        W_d = 3π · d_p^L · ν_L · (1-ε_p) · Cd(Re_p, ε_p)

        Args:
            dp:     Effective particle diameter [m].
            u_slip: Slip velocity (u_solid - u_fluid) [m/s].
            svf:    Solid volume fraction ε_p ∈ [0, 1).

        Returns:
            Dimensionless lattice weight W_d.
        """
        u_mag = tm.length(u_slip)
        Re_p = (1.0 - svf) * self.rho0 * dp * u_mag / self.mu
        Cd = self._compute_tenneti_Cd(Re_p, svf)
        dp_L = dp / self.unit.dx
        Wd = 3.0 * tm.pi * dp_L * self.nuLu * (1.0 - svf) * Cd
        return Wd

    @ti.func
    def _compute_tenneti_Cd(self, Re_p: float, svf: float) -> float:
        """
        Tenneti drag coefficient for dense suspensions.

        Cd(Re_p, ε_p) = (1-ε_p) · [Cd0/(1-ε_p)³ + A(ε_p) + B(Re_p, ε_p)]

        where:
          Cd0  = 1 + 0.15·Re_p^0.687                    (single-particle drag)
          A    = 5.81·ε_p/(1-ε_p)³ + 0.48·ε_p^(1/3)/(1-ε_p)⁴   (static correction)
          B    = ε_p³·Re_p·[0.95 + 0.61·ε_p³/(1-ε_p)²]          (dynamic correction)

        Args:
            Re_p: Particle Reynolds number.
            svf:  Solid volume fraction ε_p ∈ [0, 1).

        Returns:
            Drag coefficient Cd (dimensionless).
        """
        Cd = 0.0
        fvf = 1.0 - svf  # fluid volume fraction
        if fvf > 1e-9:
            Cd0 = 1.0 + 0.15 * tm.pow(Re_p, 0.687)
            A_eps = (5.81 * svf / (fvf ** 3)
                     + 0.48 * tm.pow(svf, 1.0 / 3.0) / (fvf ** 4))
            svf3 = svf ** 3
            B_eps = svf3 * Re_p * (0.95 + 0.61 * svf3 / (fvf ** 2))
            Cd = fvf * (Cd0 / (fvf ** 3) + A_eps + B_eps)
        return Cd

    # =====================================================================
    # Collision Step
    # =====================================================================

    @ti.kernel
    def collide(self):
        """
        Hybrid collision step with per-cell dispatch.

        Dispatch logic (priority order):
          1. Coarse + Fine  → collide_hybrid  (both present)
          2. Coarse only    → collide_coarse   (PSC momentum exchange)
          3. Fine only      → collide_fine     (IBM weighted collision)
          4. Neither        → collide_fluid    (standard BGK)
        """
        # Reset hydrodynamic force/torque accumulators
        self.hydroforce.fill(0.0)
        self.hydrotorque.fill(0.0)

        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD
                                   | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                continue

            self.computeOmega(i, j, k)
            self.compute_feq(i, j, k)

            has_coarse = self.coarse_volfrac[i, j, k] > 0.0
            has_fine = self.fine_volfrac[i, j, k] > 0.0

            if has_coarse and has_fine:
                self.collide_hybrid(i, j, k)
            elif has_coarse:
                self.collide_coarse(i, j, k)
            elif has_fine:
                self.collide_fine(i, j, k)
            else:
                self.collide_fluid(i, j, k)

    # ------------------------------------------------------------------
    # Collision Sub-routines
    # ------------------------------------------------------------------

    @ti.func
    def collide_fluid(self, i: int, j: int, k: int):
        """Standard single-relaxation-time BGK collision."""
        for q in ti.static(range(HybridLattice3D.Q)):
            self.fpc[i, j, k][q] = (
                (1.0 - self.omega[i, j, k]) * self.f[i, j, k][q]
                + self.omega[i, j, k] * self.feq[i, j, k][q]
            )

    @ti.func
    def collide_coarse(self, i: int, j: int, k: int):
        """
        PSC collision for resolved (coarse) particles.

        f_post = f + B·Ω_s + (1-B)·Ω_f

        where Ω_s = bounce-back operator, Ω_f = BGK fluid operator.
        Momentum exchange is accumulated to hydroforce.
        """
        self.compute_feq_solid_coarse(i, j, k)
        B = self.coarse_weight[i, j, k]

        for q in range(HybridLattice3D.Q):
            Omega_s = (self.f[i, j, k][HybridLattice3D.qinv[q]]
                       - self.feq[i, j, k][HybridLattice3D.qinv[q]]
                       + self.coarse_feqsolid[i, j, k][q]
                       - self.f[i, j, k][q])
            Omega_f = -self.omega[i, j, k] * (self.f[i, j, k][q] - self.feq[i, j, k][q])

            self.fpc[i, j, k][q] = (
                self.f[i, j, k][q] + B * Omega_s + (1.0 - B) * Omega_f
            )

            # Momentum exchange: F = -B · Ω_s · c_q   [lattice units]
            self.hydroforce[i, j, k] -= B * Omega_s * HybridLattice3D.c[q]

    @ti.func
    def collide_fine(self, i: int, j: int, k: int):
        """
        IBM weighted collision for unresolved (fine) particles.

        f_post = f + β·Ω_s + (1-β)·Ω_f

        Weight β derived from Tenneti drag model.
        """
        self.compute_feq_solid_fine(i, j, k)
        beta = self.fine_weight[i, j, k]

        for q in range(HybridLattice3D.Q):
            Omega_s = (self.f[i, j, k][HybridLattice3D.qinv[q]]
                       - self.feq[i, j, k][HybridLattice3D.qinv[q]]
                       + self.fine_feqsolid[i, j, k][q]
                       - self.f[i, j, k][q])
            Omega_f = -self.omega[i, j, k] * (self.f[i, j, k][q] - self.feq[i, j, k][q])

            self.fpc[i, j, k][q] = (
                self.f[i, j, k][q] + beta * Omega_s + (1.0 - beta) * Omega_f
            )

    @ti.func
    def collide_hybrid(self, i: int, j: int, k: int):
        """
        Hybrid collision for cells containing both coarse and fine particles.

        f_post = f + β·Ω_s¹ + B·Ω_s² + (1-β-B)·Ω_f

        where Ω_s¹ = fine bounce-back, Ω_s² = coarse bounce-back.
        Weights are clamped: B + β ≤ 1 for stability.
        Coarse momentum exchange is accumulated to hydroforce.
        """
        self.compute_feq_solid_fine(i, j, k)
        self.compute_feq_solid_coarse(i, j, k)

        B = self.coarse_weight[i, j, k]
        beta = self.fine_weight[i, j, k]

        for q in range(HybridLattice3D.Q):
            # Fine-particle bounce-back operator
            Omega_s1 = (self.f[i, j, k][HybridLattice3D.qinv[q]]
                        - self.feq[i, j, k][HybridLattice3D.qinv[q]]
                        + self.fine_feqsolid[i, j, k][q]
                        - self.f[i, j, k][q])

            # Coarse-particle bounce-back operator
            Omega_s2 = (self.f[i, j, k][HybridLattice3D.qinv[q]]
                        - self.feq[i, j, k][HybridLattice3D.qinv[q]]
                        + self.coarse_feqsolid[i, j, k][q]
                        - self.f[i, j, k][q])

            # Fluid BGK operator
            Omega_f = -self.omega[i, j, k] * (self.f[i, j, k][q] - self.feq[i, j, k][q])

            # Stability clamp
            if B + beta > 1.0:
                beta = 1.0 - B

            self.fpc[i, j, k][q] = (
                self.f[i, j, k][q]
                + beta * Omega_s1
                + B * Omega_s2
                + (1.0 - B - beta) * Omega_f
            )

            # Momentum exchange from coarse phase only
            self.hydroforce[i, j, k] -= B * Omega_s2 * HybridLattice3D.c[q]

    # =====================================================================
    # Equilibrium at Solid Velocity
    # =====================================================================

    @ti.func
    def compute_feq_solid_coarse(self, i: int, j: int, k: int):
        """Compute feq at coarse-grain solid velocity."""
        u = self.coarse_velsolid[i, j, k]
        uv = tm.dot(u, u)
        for q in ti.static(range(HybridLattice3D.Q)):
            cu = tm.dot(HybridLattice3D.c[q], u)
            self.coarse_feqsolid[i, j, k][q] = (
                HybridLattice3D.w[q] * self.rho[i, j, k]
                * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uv)
            )

    @ti.func
    def compute_feq_solid_fine(self, i: int, j: int, k: int):
        """Compute feq at fine-grain solid velocity."""
        u = self.fine_velsolid[i, j, k]
        uv = tm.dot(u, u)
        for q in ti.static(range(HybridLattice3D.Q)):
            cu = tm.dot(HybridLattice3D.c[q], u)
            self.fine_feqsolid[i, j, k][q] = (
                HybridLattice3D.w[q] * self.rho[i, j, k]
                * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uv)
            )

    # =====================================================================
    # Lattice → Grains  —  Force Transfer
    # =====================================================================

    @ti.kernel
    def lattice2grains(self):
        """
        Unified force transfer: Lattice → Grains.

        Call order in main loop:
          1. collide()
          2. stream()
          3. apply_bc()
          4. compute_macro()   (updates self.vel)
          5. lattice2grains()  (THIS function)

        Handles both:
          - Coarse: PSC momentum exchange accumulated during collision.
          - Fine:   Tenneti drag force via IBM interpolation.
        """
        # Zero all fluid forces and torques on every grain
        self.dem.gf.force_fluid.fill(0.0)
        self.dem.gf.moment_fluid.fill(0.0)

        # Coarse particles: PSC momentum exchange
        for gid in range(self.dem.gf.shape[0]):
            if 2.0 * self.dem.gf[gid].radius <= self.unit.dx:
                continue
            self._transfer_coarse_force(gid)

        # Fine particles: Tenneti drag via IBM interpolation
        for gid in range(self.dem.gf.shape[0]):
            if 2.0 * self.dem.gf[gid].radius > self.unit.dx:
                continue
            self._transfer_fine_force(gid)

    @ti.func
    def _transfer_coarse_force(self, gid: int):
        """
        Accumulate PSC momentum exchange for a single coarse grain.

        Converts lattice-unit hydroforce to physical force:
          F_phys = F_latt · ρ · dx⁴ / dt²
        """
        xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin
              + 0.5 * self.unit.dx) / self.unit.dx
        yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin
              + 0.5 * self.unit.dx) / self.unit.dx
        zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin
              + 0.5 * self.unit.dx) / self.unit.dx
        r = self.dem.gf[gid].radius / self.unit.dx

        x0 = ti.max(0, int(xc - r))
        x1 = ti.min(self.Nx, int(xc + r + 2))
        y0 = ti.max(0, int(yc - r))
        y1 = ti.min(self.Ny, int(yc + r + 2))
        z0 = ti.max(0, int(zc - r))
        z1 = ti.min(self.Nz, int(zc + r + 2))

        for i in range(x0, x1):
            for j in range(y0, y1):
                for k in range(z0, z1):
                    if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD
                                           | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                        continue

                    if self.coarse_volfrac[i, j, k] > 0.0 and self.coarse_id[i, j, k] == gid:
                        Ff = (self.hydroforce[i, j, k]
                              * self.unit.rho * self.unit.dx ** 4 / self.unit.dt ** 2)
                        self.dem.gf[gid].force_fluid += Ff

                        # Torque: τ = (x_node - x_grain) × F   (reversed sign convention)
                        r_vec = (Vector3(xc, yc, zc) - Vector3(i, j, k)) * self.unit.dx
                        self.dem.gf[gid].moment_fluid += -tm.cross(r_vec, Ff)

    @ti.func
    def _transfer_fine_force(self, gid: int):
        """
        Calculate Tenneti drag force for a fine grain via IBM interpolation.

        Steps:
          1. Interpolate fluid velocity to particle location (kernel-weighted).
          2. Compute average fluid volume fraction at particle location.
          3. Apply Tenneti drag model: F_d = -3π·d_p·μ·ε_f·Cd·u_slip.
        """
        xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin
              + 0.5 * self.unit.dx) / self.unit.dx
        yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin
              + 0.5 * self.unit.dx) / self.unit.dx
        zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin
              + 0.5 * self.unit.dx) / self.unit.dx

        support = 1.5  # threedelta kernel support radius
        x0 = ti.max(0, ti.cast(xc - support, ti.i32))
        x1 = ti.min(self.Nx, ti.cast(xc + support + 1, ti.i32))
        y0 = ti.max(0, ti.cast(yc - support, ti.i32))
        y1 = ti.min(self.Ny, ti.cast(yc + support + 1, ti.i32))
        z0 = ti.max(0, ti.cast(zc - support, ti.i32))
        z1 = ti.min(self.Nz, ti.cast(zc + support + 1, ti.i32))

        # Accumulators for IBM interpolation
        vel_wsum = Vector3(0.0, 0.0, 0.0)
        eps_sum = 0.0
        w_total = 0.0
        n_lattice = 0

        for i in range(x0, x1):
            for j in range(y0, y1):
                for k in range(z0, z1):
                    if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD
                                           | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                        continue

                    w_ij = self._kernel_with_mirror(xc, yc, zc, i, j, k)
                    if w_ij < 0.0:
                        continue

                    vel_wsum += self.vel[i, j, k] * w_ij
                    eps_sum += (1.0 - self.fine_volfrac[i, j, k]
                                - self.coarse_volfrac[i, j, k])
                    w_total += w_ij
                    n_lattice += 1

        # Recover fluid quantities at particle location
        fluid_vel = Vector3(0.0, 0.0, 0.0)
        eps_fluid = 0.0

        if w_total > 1e-15:
            fluid_vel = vel_wsum / w_total * self.unit.dx / self.unit.dt
        if n_lattice > 0:
            eps_fluid = eps_sum / float(n_lattice)

        # Apply Tenneti drag
        eps_p = 1.0 - eps_fluid
        if eps_p > 1e-15:
            d_p = 2.0 * self.dem.gf[gid].radius
            u_slip = self.dem.gf[gid].velocity - fluid_vel
            self.dem.gf[gid].force_fluid += self._compute_drag_force(d_p, u_slip, eps_p)

    @ti.func
    def _compute_drag_force(self, dp: float, u_slip: Vector3, svf: float) -> Vector3:
        """
        Tenneti drag force vector [N].

        F_d = -3π · d_p · μ · (1-ε_p) · Cd(Re_p, ε_p) · u_slip

        Args:
            dp:     Particle diameter [m].
            u_slip: Slip velocity u_p - u_f [m/s].
            svf:    Solid volume fraction ε_p ∈ [0, 1).

        Returns:
            Drag force vector [N].
        """
        u_mag = tm.length(u_slip)
        Re_p = (1.0 - svf) * self.rho0 * dp * u_mag / self.mu
        Cd = self._compute_tenneti_Cd(Re_p, svf)
        F_drag = -3.0 * tm.pi * dp * self.mu * (1.0 - svf) * Cd * u_slip
        return F_drag

    # =====================================================================
    # IBM Kernel Utilities
    # =====================================================================

    @ti.func
    def _threedelta(self, r: float) -> float:
        """
        Peskin 3-point delta function (radial form).

        Support: |r| ≤ 1.5

        Region 1 (0 ≤ r < 0.5):
            φ(r) = [1 + √(-3r² + 1)] / 3
        Region 2 (0.5 ≤ r ≤ 1.5):
            φ(r) = [5 - 3r - √(-3(1-r)² + 1)] / 6
        Region 3 (r > 1.5):
            φ(r) = 0
        """
        a = 0.0
        if r < 0.5:
            x = -3.0 * r * r + 1.0
            a = (1.0 + ti.sqrt(x)) / 3.0
        elif r <= 1.5:
            x = -3.0 * (1.0 - r) ** 2 + 1.0
            a = (5.0 - 3.0 * r - ti.sqrt(x)) / 6.0
        return a

    @ti.func
    def _kernel_with_mirror(self, xc: float, yc: float, zc: float,
                            ii: int, jj: int, kk: int) -> float:
        """
        Mirror-extended kernel W_bar = W(x_p) + W(x'_p) at lattice node (ii,jj,kk).

        Implements boundary treatment Eq.(22) of Zhu et al. (2026):
        The truncated kernel lobe is folded back into the domain by adding the
        contribution from the mirror particle x'_p, reflected about the nearest
        domain wall. Triggered when particle centre is within 1.5 lu of a wall.

        All kernel evaluations (primary and mirror) use the unified radial form
        threedelta(dist) for consistency.

        Args:
            xc, yc, zc: Particle centre in lattice coordinates.
            ii, jj, kk: Target lattice node indices.

        Returns:
            Mirror-corrected kernel weight W_bar.
        """
        support = 1.5

        # W(x_p) — primary contribution
        dist = ti.sqrt((xc - ii) ** 2 + (yc - jj) ** 2 + (zc - kk) ** 2)
        w_primary = self._threedelta(dist)

        # W(x'_p) — mirror contributions
        w_mirror = 0.0

        # -- x walls --
        if xc < support:
            xc_mir = -xc
            dist = ti.sqrt((xc_mir - ii) ** 2 + (yc - jj) ** 2 + (zc - kk) ** 2)
            w_mirror += self._threedelta(dist)
        if xc > float(self.Nx - 1) - support:
            xc_mir = 2.0 * float(self.Nx - 1) - xc
            dist = ti.sqrt((xc_mir - ii) ** 2 + (yc - jj) ** 2 + (zc - kk) ** 2)
            w_mirror += self._threedelta(dist)

        # -- y walls --
        if yc < support:
            yc_mir = -yc
            dist = ti.sqrt((xc - ii) ** 2 + (yc_mir - jj) ** 2 + (zc - kk) ** 2)
            w_mirror += self._threedelta(dist)
        if yc > float(self.Ny - 1) - support:
            yc_mir = 2.0 * float(self.Ny - 1) - yc
            dist = ti.sqrt((xc - ii) ** 2 + (yc_mir - jj) ** 2 + (zc - kk) ** 2)
            w_mirror += self._threedelta(dist)

        # -- z walls --
        if zc < support:
            zc_mir = -zc
            dist = ti.sqrt((xc - ii) ** 2 + (yc - jj) ** 2 + (zc_mir - kk) ** 2)
            w_mirror += self._threedelta(dist)
        if zc > float(self.Nz - 1) - support:
            zc_mir = 2.0 * float(self.Nz - 1) - zc
            dist = ti.sqrt((xc - ii) ** 2 + (yc - jj) ** 2 + (zc_mir - kk) ** 2)
            w_mirror += self._threedelta(dist)

        return w_primary + w_mirror

    # =====================================================================
    # High-Level Interface
    # =====================================================================

    def initialize_complete(self):
        """Complete initialisation sequence for hybrid coupling."""
        self.initialize()
        self.map_coarse_grains()
        self.map_fine_grains()
        self.compute_fine_weights()

    def update_coupling(self):
        """
        Update coupling fields each time step.

        Order matters:
          1. map_coarse_grains()  — coarse volfrac needed by fine mapping
          2. map_fine_grains()    — uses coarse_volfrac for availability correction
          3. compute_fine_weights() — uses fine_volfrac + fluid velocity
          4. lattice2grains()     — transfers forces back to DEM particles
        """
        self.map_coarse_grains()
        self.map_fine_grains()
        self.compute_fine_weights()
        self.lattice2grains()
