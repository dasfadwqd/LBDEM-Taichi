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
    - `fine_volfrac`: Contribution from unresolved particles (IBM).
* Collision Dispatch:
    - Pure Fluid: Standard BGK.
    - Coarse Cell: PSC bounce-back (Priority if mixed).
    - Fine Cell: IBM weighted collision.
    - Hybrid Cell: Weighted combination of PSC and IBM operators.
* Force Transfer:
    - Unified `lattice2grains()` kernel handles both particle types.
    - Forces are zeroed at the start of this kernel to ensure consistency.

References
----------
* PSC Method: Noble & Torczynski, Int. J. Mod. Phys. C 9 (1998) 1189-1201
* Tenneti Drag: Tenneti et al., Int. J. Multiphase Flow 37 (2011) 1072-1092
* Equiv. IBM: Wang et al., Chem. Eng. J. (2023) 142898
"""

import taichi as ti
import taichi.math as tm

from src.lbm3d.lbm_solver3d import BasicLattice3D
from src.lbm3d.lbmutils import CellType
from src.dem3d.demsolver import DEMSolver

# ---------------------------------------------------------------------------
# Type Aliases & Constants
# ---------------------------------------------------------------------------
Vector3 = ti.types.vector(3, float)

class HybridLattice3D(BasicLattice3D):
    """
    3-D LBM lattice with hybrid resolved/unresolved FSI coupling.

    Particle Classification (Runtime)
    ---------------------------------
    For each DEM grain `gid`:
        - Coarse (Resolved): 2 * radius > dx  → PSC Method
        - Fine (Unresolved): 2 * radius ≤ dx  → IBM/Tenneti Method

    This allows polydisperse simulations where each grain is handled by the
    most appropriate coupling strategy automatically.
    """

    # D3Q19 inverse direction table (static for kernel inlining)
    QINV_STATIC = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17)

    def __init__(
        self,
        Nx: int, Ny: int, Nz: int,
        omega: float,
        dx: float, dt: float,
        rho: float,
        dem_solver: DEMSolver,  # Renamed from 'demslover' for clarity
    ):
        super().__init__(Nx, Ny, Nz, omega, dx, dt, rho)
        shape = (Nx, Ny, Nz)

        # ------------------------------------------------------------------
        # 1. Fluid Properties (Physical & Lattice Units)
        # ------------------------------------------------------------------
        self.rho0 = rho
        self.omega0 = omega
        # Lattice kinematic viscosity: nu_L = (1/omega - 0.5) * c_s^2 (c_s^2 = 1/3)
        self.nuLu = (1.0 / omega - 0.5) / 3.0
        # Physical kinematic viscosity: nu_P = nu_L * dx^2 / dt
        self.nu = self.nuLu * (dx ** 2) / dt
        # Dynamic viscosity: mu = rho * nu_P
        self.mu = rho * self.nu

        # ------------------------------------------------------------------
        # 2. Coupling Fields (Volume Fractions & Velocities)
        # ------------------------------------------------------------------
        # Solid volume fractions (split by particle class)
        self.coarse_volfrac = ti.field(float, shape=shape)  # Resolved (PSC)
        self.fine_volfrac = ti.field(float, shape=shape)    # Unresolved (IBM)

        # Solid velocity fields (at lattice nodes)
        self.coarse_velsolid = ti.Vector.field(self.D, float, shape=shape)
        self.fine_velsolid = ti.Vector.field(self.D, float, shape=shape)

        # Weighting coefficients for collision operators
        self.coarse_weight = ti.field(float, shape=shape)   # B (PSC)
        self.fine_weight = ti.field(float, shape=shape)     # beta (Tenneti)

        # Equilibrium distributions computed at solid velocity
        self.coarse_feqsolid = ti.Vector.field(self.Q, float, shape=shape)
        self.fine_feqsolid = ti.Vector.field(self.Q, float, shape=shape)

        # ------------------------------------------------------------------
        # 3. Force & Momentum Exchange Fields
        # ------------------------------------------------------------------
        # Momentum-exchange force/torque (Coarse particles only, per cell)
        self.hydroforce = ti.Vector.field(self.D, float, shape=shape)
        self.hydrotorque = ti.Vector.field(self.D, float, shape=shape)

        # Grain ID map for coarse particles (-1 indicates no coarse grain)
        self.coarse_id = ti.field(int, shape=shape)

        # Temporary accumulators for IBM (Fine particles)
        self.fine_velsum = ti.Vector.field(self.D, float, shape=shape)
        self.fine_weightsum = ti.field(float, shape=shape)

        # ------------------------------------------------------------------
        # 4. DEM Solver Reference
        # ------------------------------------------------------------------
        self.dem = dem_solver

    # ======================================================================
    # Initialization
    # ======================================================================

    @ti.kernel
    def initialize(self):
        """
        Initialize distribution functions to equilibrium and map grains.
        """
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # Skip solid/boundary cells
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            self.compute_feq(i, j, k)
            for q in ti.static(range(HybridLattice3D.Q)):
                self.f[i, j, k][q] = self.feq[i, j, k][q]

    # ======================================================================
    # DEM → Lattice Mapping
    # ======================================================================

    @ti.kernel
    def map_coarse_grains(self):
        """
        Map resolved (coarse) grains using the PSC solid-fraction method.

        Uses a 5×5×5 sub-cell decomposition to estimate solid volume fraction (ε)
        for partially covered cells. Fully covered cells get ε = 1.
        Only grains with diameter > dx contribute here.
        """
        # Reset fields
        self.coarse_id.fill(-1)
        self.coarse_volfrac.fill(0.0)
        self.coarse_weight.fill(0.0)
        self.coarse_velsolid.fill(0.0)

        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            for gid in range(self.dem.gf.shape[0]):
                # Filter: Skip fine particles
                if 2.0 * self.dem.gf[gid].radius <= self.unit.dx:
                    continue

                # Grain center in lattice coordinates
                xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
                yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
                zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx

                # Effective radius (shrunk by 0.5dx to treat boundary cells as partial)
                r = (self.dem.gf[gid].radius - 0.5 * self.unit.dx) / self.unit.dx

                dist = ti.sqrt((xc - i)**2 + (yc - j)**2 + (zc - k)**2)

                # Case 1: Fully Outside
                if dist >= r + 0.5 * ti.sqrt(3.0):
                    continue

                # Case 2: Fully Inside
                if dist <= r - 0.5 * ti.sqrt(3.0):
                    self.coarse_id[i, j, k] = gid
                    self.coarse_volfrac[i, j, k] = 1.0
                    self.set_coarse_weight(i, j, k)
                    self.set_coarse_velsolid(i, j, k, gid, xc, yc, zc)
                    break  # One grain dominates per cell

                # Case 3: Partially Covered (5^3 sub-cell integration)
                cnt = 0
                for si in range(5):
                    for sj in range(5):
                        for sk in range(5):
                            # Sub-cell center coordinates
                            sub_x = i - 0.4 + 0.2 * si
                            sub_y = j - 0.4 + 0.2 * sj
                            sub_z = k - 0.4 + 0.2 * sk

                            d2 = ti.sqrt((xc - sub_x)**2 + (yc - sub_y)**2 + (zc - sub_z)**2)
                            if d2 < r:
                                cnt += 1

                eps = cnt / 125.0

                # Keep the grain with the largest coverage per cell
                if eps > self.coarse_volfrac[i, j, k]:
                    self.coarse_id[i, j, k] = gid
                    self.coarse_volfrac[i, j, k] = eps
                    self.set_coarse_weight(i, j, k)
                    self.set_coarse_velsolid(i, j, k, gid, xc, yc, zc)

    @ti.func
    def set_coarse_weight(self, i: int, j: int, k: int):
        """
        Calculate PSC weighting coefficient B.
        Formula: B = ε(τ-½) / [(1-ε)+(τ-½)]
        """
        eps = self.coarse_volfrac[i, j, k]
        tau_m = 1.0 / self.omega[i, j, k] - 0.5
        self.coarse_weight[i, j, k] = (eps * tau_m) / ((1.0 - eps) + tau_m)

    @ti.func
    def set_coarse_velsolid(self, i: int, j: int, k: int, gid: int, xc: float, yc: float, zc: float):
        """
        Calculate solid velocity at node (i,j,k) for coarse grain.
        Formula: u_s = u_trans + omega x r (converted to lattice units)
        """
        r_vec = Vector3(i, j, k) - Vector3(xc, yc, zc)
        omega_cross_r = tm.cross(self.dem.gf[gid].omega, r_vec * self.unit.dx)

        self.coarse_velsolid[i, j, k] = (
            (self.dem.gf[gid].velocity + omega_cross_r) * self.unit.dt / self.unit.dx
        )

    # ------------------------------------------------------------------

    @ti.kernel
    def map_fine_grains(self):
        """
        Map unresolved (fine) grains using a regularised IBM kernel.

        Uses the 3-point delta function with support radius 1.5 dx.
        Includes boundary correction for weight redistribution.
        Only grains with diameter ≤ dx contribute here.
        """
        self.fine_volfrac.fill(0.0)
        self.fine_velsum.fill(0.0)
        self.fine_weightsum.fill(0.0)

        V_lattice = self.unit.dx ** 3

        for gid in range(self.dem.gf.shape[0]):
            # Filter: Skip coarse particles
            if 2.0 * self.dem.gf[gid].radius > self.unit.dx:
                continue

            # Grain center in lattice coordinates
            xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx)

            V_grain = 4.0 / 3.0 * tm.pi * self.dem.gf[gid].radius ** 3
            vel_lattice = self.dem.gf[gid].velocity * self.unit.dt / self.unit.dx
            support_radius = 1.5

            # Determine search bounds
            i_min = ti.max(0, ti.cast(xc - support_radius, ti.i32))
            i_max = ti.min(self.Nx, ti.cast(xc + support_radius + 1, ti.i32))
            j_min = ti.max(0, ti.cast(yc - support_radius, ti.i32))
            j_max = ti.min(self.Ny, ti.cast(yc + support_radius + 1, ti.i32))
            k_min = ti.max(0, ti.cast(zc - support_radius, ti.i32))
            k_max = ti.min(self.Nz, ti.cast(zc + support_radius + 1, ti.i32))

            # Pass 1: Calculate total vs valid weight (for boundary correction)
            total_w = 0.0
            valid_w = 0.0
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    for k in range(k_min, k_max):
                        dist_x = ti.abs(xc - i)
                        dist_y = ti.abs(yc - j)
                        dist_z = ti.abs(zc - k)

                        w = self._threedelta(dist_x) * self._threedelta(dist_y) * self._threedelta(dist_z)
                        if w <= 0.0:
                            continue

                        total_w += w
                        if not (self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP)):
                            valid_w += w

            if valid_w < 1e-12:
                continue

            corr = total_w / valid_w  # Redistribution factor

            # Pass 2: Accumulate into lattice fields
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    for k in range(k_min, k_max):
                        if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                            continue

                        dist_x = ti.abs(xc - i)
                        dist_y = ti.abs(yc - j)
                        dist_z = ti.abs(zc - k)

                        w = self._threedelta(dist_x) * self._threedelta(dist_y) * self._threedelta(dist_z)
                        if w <= 0.0:
                            continue

                        w_c = w * corr
                        ti.atomic_add(self.fine_volfrac[i, j, k], w_c * V_grain / V_lattice)
                        ti.atomic_add(self.fine_velsum[i, j, k], w_c * vel_lattice)
                        ti.atomic_add(self.fine_weightsum[i, j, k], w_c)

        # Normalise velocity and clamp volume fraction
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.fine_volfrac[i, j, k] >= 1.0:
                self.fine_volfrac[i, j, k] = 0.9999
                # Warning printed inside kernel for debugging
                print("Warning: fine particles volfrac[{}, {}, {}] >= 1.0".format(i, j, k))

            if self.fine_weightsum[i, j, k] > 1e-10:
                self.fine_velsolid[i, j, k] = self.fine_velsum[i, j, k] / self.fine_weightsum[i, j, k]

    # ======================================================================
    # Collision Step
    # ======================================================================

    @ti.kernel
    def collide(self):
        """
        Hybrid collision step with per-cell dispatch.

        Priority Logic:
        1. Coarse (PSC) if coarse_volfrac > 0
        2. Fine (IBM) if fine_volfrac > 0
        3. Hybrid if both > 0 (Coarse physics takes priority, Fine adds drag)
        4. Pure Fluid otherwise
        """
        # Reset hydrodynamic force/torque accumulators
        self.hydroforce.fill(0.0)
        self.hydrotorque.fill(0.0)

        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            self.compute_feq(i, j, k)

            has_coarse = self.coarse_volfrac[i, j, k] > 0.0
            has_fine = self.fine_volfrac[i, j, k] > 0.0

            if has_coarse and not has_fine:
                # --- Resolved particle: PSC momentum-exchange ---
                self.collide_coarse(i, j, k)

            elif has_fine and not has_coarse:
                # --- Unresolved particle: IBM weighted collision ---
                self.collide_fine(i, j, k)

            elif has_coarse and has_fine:
                # --- Hybrid cell: Combined operator ---
                self.collide_hybrid(i, j, k)

            else:
                # --- Pure fluid: Standard BGK ---
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
        Formula: f_post = f + B·Ω_s + (1-B)·Ω_f
        Accumulates hydrodynamic force via momentum exchange.
        """
        self.compute_feq_solid_coarse(i, j, k)
        B = self.coarse_weight[i, j, k]

        for q in range(HybridLattice3D.Q):
            q_inv = HybridLattice3D.QINV_STATIC[q]

            # Solid collision operator (Bounce-back with solid feq)
            Omega_s = (
                self.f[i, j, k][q_inv]
                - self.feq[i, j, k][q_inv]
                + self.coarse_feqsolid[i, j, k][q]
                - self.f[i, j, k][q]
            )
            # Fluid collision operator (BGK)
            Omega_f = -self.omega[i, j, k] * (self.f[i, j, k][q] - self.feq[i, j, k][q])

            self.fpc[i, j, k][q] = self.f[i, j, k][q] + B * Omega_s + (1.0 - B) * Omega_f

            # Momentum exchange: F = -B · Ω_s · c (lattice units)
            self.hydroforce[i, j, k] -= B * Omega_s * HybridLattice3D.c[q]

    @ti.func
    def collide_fine(self, i: int, j: int, k: int):
        """
        IBM-style weighted collision for unresolved (fine) particles.
        Formula: f_post = f + β·Ω_s + (1-β)·Ω_f
        Weight β derived from Tenneti drag model.
        """
        self.compute_feq_solid_fine(i, j, k)
        beta = self.fine_weight[i, j, k]

        for q in ti.static(range(HybridLattice3D.Q)):
            q_inv = ti.static(HybridLattice3D.QINV_STATIC[q])

            Omega_s = (
                self.f[i, j, k][q_inv]
                - self.feq[i, j, k][q_inv]
                + self.fine_feqsolid[i, j, k][q]
                - self.f[i, j, k][q]
            )
            Omega_f = -self.omega[i, j, k] * (self.f[i, j, k][q] - self.feq[i, j, k][q])

            self.fpc[i, j, k][q] = self.f[i, j, k][q] + beta * Omega_s + (1.0 - beta) * Omega_f

    @ti.func
    def collide_hybrid(self, i: int, j: int, k: int):
        """
        Hybrid collision for cells containing both coarse and fine contributions.
        Formula: f_post = f + β·Ω_s(fine) + B·Ω_s(coarse) + (1-β-B)·Ω_f
        Note: Weights are clamped to ensure stability (B + β <= 1).
        """
        self.compute_feq_solid_fine(i, j, k)
        self.compute_feq_solid_coarse(i, j, k)

        B = self.coarse_weight[i, j, k]
        beta = self.fine_weight[i, j, k]

        for q in range(HybridLattice3D.Q):
            q_inv = HybridLattice3D.QINV_STATIC[q]

            # Fine particle operator
            Omega_s1 = (
                self.f[i, j, k][q_inv]
                - self.feq[i, j, k][q_inv]
                + self.fine_feqsolid[i, j, k][q]
                - self.f[i, j, k][q]
            )
            # Coarse particle operator
            Omega_s2 = (
                self.f[i, j, k][q_inv]
                - self.feq[i, j, k][q_inv]
                + self.coarse_feqsolid[i, j, k][q]
                - self.f[i, j, k][q]
            )
            # Fluid operator
            Omega_f = -self.omega[i, j, k] * (self.f[i, j, k][q] - self.feq[i, j, k][q])

            # Safety clamp for weights
            if B + beta > 1.0:
                beta = 1.0 - B

            self.fpc[i, j, k][q] = (
                self.f[i, j, k][q] + beta * Omega_s1 + B * Omega_s2 + (1.0 - B - beta) * Omega_f
            )

            # Momentum exchange (Coarse only contributes to lattice force here)
            self.hydroforce[i, j, k] -= B * Omega_s2 * HybridLattice3D.c[q]

    # ------------------------------------------------------------------
    # Equilibrium Helpers
    # ------------------------------------------------------------------

    @ti.func
    def compute_feq_solid_coarse(self, i: int, j: int, k: int):
        """Compute feq at coarse-grain solid velocity."""
        u = self.coarse_velsolid[i, j, k]
        uv = tm.dot(u, u)
        for q in range(HybridLattice3D.Q):
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

    # ======================================================================
    # Drag Model (Tenneti)
    # ======================================================================

    @ti.kernel
    def compute_fine_weights(self):
        """
        Compute per-cell IBM weight coefficient using Tenneti drag model.
        Must be called after mapping fine grains.
        """
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            svf = self.fine_volfrac[i, j, k]
            if svf > 0.0:
                V_lattice = self.unit.dx ** 3
                # Effective particle radius from volume fraction
                R_eff = ti.pow(3.0 * V_lattice * svf / (4.0 * tm.pi), 1.0 / 3.0)
                d_eff = 2.0 * R_eff

                # Slip velocity in physical units
                v_slip = ((self.fine_velsolid[i, j, k] - self.vel[i, j, k])
                          * self.unit.dx / self.unit.dt)

                self.fine_weight[i, j, k] = self._tenneti_weight(d_eff, v_slip, svf)

                if self.fine_weight[i, j, k] > 1.0:
                    print("Warning:fine weight[{}, {}, {}] > 1.0".format(i, j, k))

    @ti.func
    def _tenneti_weight(self, dp: float, u_slip: Vector3, svf: float) -> float:
        """
        Calculate Tenneti weight W_d (dimensionless, lattice units).
        W_d = 3π d_p^L ν_L (1-ε) C_d
        """
        u_mag = tm.length(u_slip)
        Re_p = (1.0 - svf) * self.rho0 * dp * u_mag / self.mu

        # Compute Drag Coefficient Cd
        Cd = self._compute_tenneti_Cd(Re_p, svf)

        dp_L = dp / self.unit.dx  # Diameter in lattice units
        Wd = 3.0 * tm.pi * dp_L * self.nuLu * (1.0 - svf) * Cd
        return Wd

    @ti.func
    def _compute_tenneti_Cd(self, Re_p: float, svf: float) -> float:
        """
        Shared helper for Tenneti Drag Coefficient calculation.
        Used by both weight calculation and force calculation.
        """
        Cd = 0.0
        if (1.0 - svf) > 1e-9:
            Cd0 = 1.0 + 0.15 * tm.pow(Re_p, 0.687)

            # Static correction term A(ε_p)
            A_eps = (5.81 * svf / (1.0 - svf)**3
                     + 0.48 * tm.pow(svf, 1.0/3.0) / (1.0 - svf)**4)

            # Dynamic correction term B(Re_p, ε_p)
            svf3 = svf ** 3
            B_eps = svf3 * Re_p * (0.95 + 0.61 * svf3 / (1.0 - svf)**2)

            Cd = (1.0 - svf) * (Cd0 / (1.0 - svf)**3 + A_eps + B_eps)
        return Cd

    # ======================================================================
    # Lattice → Grains Force Transfer
    # ======================================================================

    @ti.kernel
    def lattice2grains(self):
        """
        Unified force transfer: Lattice → Grains.

        Call Order:
        1. collide()      (fills self.hydroforce)
        2. stream()
        3. apply_bc()
        4. compute_macro() (updates self.vel)
        5. lattice2grains() (THIS function)

        Handles both Coarse (PSC) and Fine (Tenneti) particles.
        """
        # Step 0: Zero all fluid forces and torques on every grain
        self.dem.gf.force_fluid.fill(0.0)
        self.dem.gf.moment_fluid.fill(0.0)

        # Step 1: Coarse particles (PSC momentum exchange)
        for gid in range(self.dem.gf.shape[0]):
            if 2.0 * self.dem.gf[gid].radius <= self.unit.dx:
                continue  # Skip fine particles
            self._transfer_coarse_force(gid)

        # Step 2: Fine particles (Tenneti drag via IBM interpolation)
        for gid in range(self.dem.gf.shape[0]):
            if 2.0 * self.dem.gf[gid].radius > self.unit.dx:
                continue  # Skip coarse particles
            self._transfer_fine_force(gid)

    @ti.func
    def _transfer_coarse_force(self, gid: int):
        """
        Accumulate PSC momentum exchange force for a single coarse grain.
        """
        # Grain center in lattice coordinates
        xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
        yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
        zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx
        r = self.dem.gf[gid].radius / self.unit.dx

        # Bounding box for the grain
        x0 = ti.max(0, int(xc - r))
        x1 = ti.min(self.Nx, int(xc + r + 2))
        y0 = ti.max(0, int(yc - r))
        y1 = ti.min(self.Ny, int(yc + r + 2))
        z0 = ti.max(0, int(zc - r))
        z1 = ti.min(self.Nz, int(zc + r + 2))

        for i in range(x0, x1):
            for j in range(y0, y1):
                for k in range(z0, z1):
                    if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                        continue

                    if self.coarse_volfrac[i, j, k] > 0.0 and self.coarse_id[i, j, k] == gid:
                        # Convert lattice-unit momentum exchange to physical force
                        # F_phys = F_latt * rho * dx^4 / dt^2
                        Ff = (self.hydroforce[i, j, k]
                              * self.unit.rho * self.unit.dx ** 4 / self.unit.dt ** 2)

                        self.dem.gf[gid].force_fluid += Ff

                        # Torque about grain centre (r_vec: grain → cell)
                        r_vec = (Vector3(xc, yc, zc) - Vector3(i, j, k)) * self.unit.dx
                        self.dem.gf[gid].moment_fluid += tm.cross(r_vec, Ff)

    @ti.func
    def _transfer_fine_force(self, gid: int):
        """
        Calculate Tenneti drag force for a single fine grain via IBM interpolation.
        """
        # Grain center in lattice coordinates
        xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
        yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
        zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx

        # Search bounds (support radius ~2 for interpolation)
        x0 = ti.max(0, int(xc - 2))
        x1 = ti.min(self.Nx, int(xc + 2))
        y0 = ti.max(0, int(yc - 2))
        y1 = ti.min(self.Ny, int(yc + 2))
        z0 = ti.max(0, int(zc - 2))
        z1 = ti.min(self.Nz, int(zc + 2))

        # IBM interpolation: accumulate weighted fluid velocity and ε
        u_fluid_particle = Vector3(0.0, 0.0, 0.0)
        volfrac_particle = 0.0

        for i in range(x0, x1):
            for j in range(y0, y1):
                for k in range(z0, z1):
                    if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                        continue

                    dist_x = ti.abs(xc - i)
                    dist_y = ti.abs(yc - j)
                    dist_z = ti.abs(zc - k)

                    w = (self._threedelta(dist_x) *
                         self._threedelta(dist_y) *
                         self._threedelta(dist_z))

                    # self.vel is in lattice units
                    u_fluid_particle += self.vel[i, j, k] * w
                    volfrac_particle += self.fine_volfrac[i, j, k] * w

        if volfrac_particle > 0.0:
            # Convert interpolated fluid velocity to physical units
            u_fluid = u_fluid_particle * self.unit.dx / self.unit.dt
            d_p = 2.0 * self.dem.gf[gid].radius
            u_slip = self.dem.gf[gid].velocity - u_fluid

            self.dem.gf[gid].force_fluid += self._tenneti_drag(d_p, u_slip, volfrac_particle)

    @ti.func
    def _tenneti_drag(self, dp: float, u_slip: Vector3, svf: float) -> Vector3:
        """
        Calculate Tenneti drag force vector in physical units [N].
        Formula: F_d = -3π d_p μ (1-ε) C_d(Re_p, ε) u_slip
        """
        u_mag = tm.length(u_slip)
        Re_p = (1.0 - svf) * self.rho0 * dp * u_mag / self.mu

        # Compute Drag Coefficient Cd (Reuses shared logic)
        Cd = self._compute_tenneti_Cd(Re_p, svf)

        # Drag Force Vector
        F_drag = -3.0 * tm.pi * dp * self.mu * (1.0 - svf) * Cd * u_slip
        return F_drag

    # ======================================================================
    # IBM Kernel Utilities
    # ======================================================================

    @ti.func
    def _threedelta(self, r: float) -> float:
        """
        Peskin 3-point delta function kernel.
        Support: |r| ≤ 1.5
        """
        a = 0.0
        if r < 0.5:
            x = -3.0 * r * r + 1.0
            a = (1.0 + ti.sqrt(x)) / 3.0
        elif r <= 1.5:
            x = -3.0 * (1.0 - r)**2 + 1.0
            a = (5.0 - 3.0 * r - ti.sqrt(x)) / 6.0
        return a

    # ======================================================================
    # High-level interface
    # ======================================================================
    # (To be implemented by the user / calling script)
    def initialize_complete(self):
        """Complete initialization sequence for coupling."""
        self.initialize()
        self.map_fine_grains()
        self.map_coarse_grains()
        self.compute_fine_weights()

    def update_coupling(self):
        """Update coupling fields at each time step."""
        self.map_fine_grains()
        self.map_coarse_grains()
        self.compute_fine_weights()
        self.lattice2grains()