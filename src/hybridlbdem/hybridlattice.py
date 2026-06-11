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

        Volume fraction correction:
            eps_fine_ij = w_norm * (1 - eps_coarse_ij) * V_grain / (D_corrected * V_lattice)
        where D_corrected = sum_j[ w_norm_ij * (1 - eps_coarse_ij) ]
        ensures volume conservation in the presence of coarse particles.

        Note: Nodes fully occupied by coarse particles (eps_coarse >= 1.0) are
        skipped entirely to avoid floating-point residuals triggering false warnings.
        """
        # ------------------------------------------------------------------ #
        # Reset fields
        # ------------------------------------------------------------------ #
        self.fine_volfrac.fill(0.0)
        self.fine_velsum.fill(0.0)
        self.fine_weightsum.fill(0.0)

        V_lattice = self.unit.dx ** 3
        support_radius = 1.5

        for gid in range(self.dem.gf.shape[0]):
            # Filter: Skip coarse particles
            if 2.0 * self.dem.gf[gid].radius > self.unit.dx:
                continue

            # Grain center in lattice coordinates
            xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx

            V_grain = 4.0 / 3.0 * tm.pi * self.dem.gf[gid].radius ** 3
            vel_lattice = self.dem.gf[gid].velocity * self.unit.dt / self.unit.dx

            # Search bounds
            i_min = ti.max(0, ti.cast(xc - support_radius, ti.i32))
            i_max = ti.min(self.Nx, ti.cast(xc + support_radius + 1, ti.i32))
            j_min = ti.max(0, ti.cast(yc - support_radius, ti.i32))
            j_max = ti.min(self.Ny, ti.cast(yc + support_radius + 1, ti.i32))
            k_min = ti.max(0, ti.cast(zc - support_radius, ti.i32))
            k_max = ti.min(self.Nz, ti.cast(zc + support_radius + 1, ti.i32))

            # ---------------------------------------------------------------- #
            # Pass 1: kernel normalisation denominator (geometry only)
            # denom = sum_j[ W_bar(x_p - x_j) ]
            # Skip nodes fully occupied by coarse particles
            # ---------------------------------------------------------------- #
            denom = 0.0
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    for k in range(k_min, k_max):
                        if self.CT[i, j, k] & (CellType.OBSTACLE |
                                               CellType.VEL_LADD |
                                               CellType.FREE_SLIP):
                            continue
                        if self.coarse_volfrac[i, j, k] >= 1.0:
                            continue
                        w_bar = self._kernel_with_mirror(xc, yc, zc, i, j, k)
                        if w_bar > 0.0:
                            denom += w_bar

            if denom < 1e-30:
                continue

            # ---------------------------------------------------------------- #
            # Pass 2: corrected denominator accounting for coarse occupation
            # D_corrected = sum_j[ w_norm_ij * (1 - eps_coarse_ij) ]
            # Skip nodes fully occupied by coarse particles
            # ---------------------------------------------------------------- #
            denom_corrected = 0.0
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    for k in range(k_min, k_max):
                        if self.CT[i, j, k] & (CellType.OBSTACLE |
                                               CellType.VEL_LADD |
                                               CellType.FREE_SLIP):
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

            # ---------------------------------------------------------------- #
            # Pass 3: accumulate corrected volume fraction and velocity
            # eps_fine_ij = w_norm * avail_ij / denom_corrected * V_grain / V_lattice
            # Skip nodes fully occupied by coarse particles
            # ---------------------------------------------------------------- #
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    for k in range(k_min, k_max):
                        if self.CT[i, j, k] & (CellType.OBSTACLE |
                                               CellType.VEL_LADD |
                                               CellType.FREE_SLIP):
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
                        ti.atomic_add(self.fine_velsum[i, j, k], vel_lattice * eps_ij)
                        ti.atomic_add(self.fine_weightsum[i, j, k], w_norm)

        # ------------------------------------------------------------------ #
        # Normalise solid velocity; clamp volume fraction
        # ------------------------------------------------------------------ #
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            avail = 1.0 - self.coarse_volfrac[i, j, k]

            if self.fine_volfrac[i, j, k] > avail:
                # Only print for physically meaningful overflow (not float residuals)
                if self.fine_volfrac[i, j, k] - avail > 1e-6:
                    print("Warning: physical overflow at ({},{},{}): fine={}, coarse={}".format(
                        i, j, k,
                        self.fine_volfrac[i, j, k],
                        self.coarse_volfrac[i, j, k]))
                self.fine_volfrac[i, j, k] = ti.max(0.0, avail - 0.01)

            if self.fine_volfrac[i, j, k] > 1e-10:
                self.fine_velsolid[i, j, k] = (
                        self.fine_velsum[i, j, k] / self.fine_volfrac[i, j, k]
                )

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
        self.compute_feq_solid_coarse(i, j, k)
        B = self.coarse_weight[i, j, k]

        for q in range(HybridLattice3D.Q):
            Omega_s = (self.f[i, j, k][HybridLattice3D.qinv[q]] - self.feq[i, j, k][HybridLattice3D.qinv[q]] +
                       self.coarse_feqsolid[i, j, k][q] - self.f[i, j, k][q])
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

        for q in range(HybridLattice3D.Q):
            Omega_s = (self.f[i, j, k][HybridLattice3D.qinv[q]] - self.feq[i, j, k][HybridLattice3D.qinv[q]] +
                       self.fine_feqsolid[i, j, k][q] - self.f[i, j, k][q])
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
            Omega_s1 = (self.f[i, j, k][HybridLattice3D.qinv[q]] - self.feq[i, j, k][HybridLattice3D.qinv[q]] +
                       self.fine_feqsolid[i, j, k][q] - self.f[i, j, k][q])


            # Coarse particle operator
            Omega_s2 = (self.f[i, j, k][HybridLattice3D.qinv[q]] - self.feq[i, j, k][HybridLattice3D.qinv[q]] +
                        self.coarse_feqsolid[i, j, k][q] - self.f[i, j, k][q])
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
            fvf = 1- self.fine_volfrac[i, j, k] -self.coarse_volfrac[i, j, k]
            if svf > 0.0 :
                V_lattice = self.unit.dx ** 3
                # Effective particle radius from volume fraction
                R_eff = ti.pow(3.0 * V_lattice * svf / (4.0 * tm.pi), 1.0 / 3.0)
                d_eff = 2.0 * R_eff

                # Slip velocity in physical units
                v_slip = ((self.fine_velsolid[i, j, k] - self.vel[i, j, k])
                          * self.unit.dx / self.unit.dt)

                self.fine_weight[i, j, k] = self._tenneti_weight(d_eff, v_slip, fvf)

                # if self.fine_weight[i, j, k] > 1.0:
                #     print("Warning:fine weight[{}, {}, {}] > 1.0".format(i, j, k))

    @ti.func
    def _tenneti_weight(self, dp: float, u_slip: Vector3, fvf: float) -> float:
        """
        Calculate Tenneti weight W_d (dimensionless, lattice units).
        fvf: fluid volume fraction (ε_f)
        W_d = 3π d_p^L ν_L ε_f C_d
        """
        u_mag = tm.length(u_slip)
        Re_p = fvf * self.rho0 * dp * u_mag / self.mu

        # Compute Drag Coefficient Cd
        Cd = self._compute_tenneti_Cd(Re_p, fvf)

        dp_L = dp / self.unit.dx  # Diameter in lattice units
        Wd = 3.0 * tm.pi * dp_L * self.nuLu * fvf * Cd
        return Wd

    @ti.func
    def _compute_tenneti_Cd(self, Re_p: float, fvf: float) -> float:
        """
        Shared helper for Tenneti Drag Coefficient calculation.
        Used by both weight calculation and force calculation.
        """
        Cd = 0.0
        if fvf > 1e-9:
            Cd0 = 1.0 + 0.15 * tm.pow(Re_p, 0.687)

            # Static correction term A(ε_p)
            A_eps = (5.81 * (1 - fvf) / fvf**3
                     + 0.48 * tm.pow(1 - fvf, 1.0/3.0) / fvf**4)

            # Dynamic correction term B(Re_p, ε_p)

            B_eps = (1 - fvf)**3 * Re_p * (0.95 + 0.61 * (1 - fvf)**3 / fvf**2)

            Cd = fvf * (Cd0 / fvf**3 + A_eps + B_eps)
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
                        self.dem.gf[gid].moment_fluid += -tm.cross(r_vec, Ff)

    @ti.func
    def _transfer_fine_force(self, gid: int):
        """
        Calculate Tenneti drag force for a single fine grain via IBM interpolation.
        """
        # Grain center in lattice coordinates
        xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
        yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
        zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx

        # Search bounds (support radius ~2.5 for interpolation)
        x0 = ti.max(0, int(xc - 2.5))
        x1 = ti.min(self.Nx, int(xc + 2.5))
        y0 = ti.max(0, int(yc - 2.5))
        y1 = ti.min(self.Ny, int(yc + 2.5))
        z0 = ti.max(0, int(zc - 2.5))
        z1 = ti.min(self.Nz, int(zc + 2.5))

        # IBM interpolation: accumulate weighted fluid velocity and ε
        vel_wsum = Vector3(0.0, 0.0, 0.0)
        volfrac_particle = 0.0
        eps_sum = 0.0
        w_total = 0.0
        n_lattice = 0



        for i in range(x0, x1):
            for j in range(y0, y1):
                for k in range(z0, z1):
                    if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                        continue

                    w_ij = self._kernel_with_mirror(xc, yc, zc, i, j, k)
                    if w_ij < 0.0:
                        continue

                    # self.vel is in lattice units
                    vel_wsum += self.vel[i, j, k] * w_ij
                    eps_sum += (1 - self.fine_volfrac[i,j,k] - self.coarse_volfrac[i,j,k])
                    w_total += w_ij
                    n_lattice += 1

        # Recover fluid quantities at particle location
        fluid_vel = Vector3(0.0, 0.0, 0.0)
        eps_fluid = 0.0

        if  w_total > 1e-15:
            fluid_vel = vel_wsum / w_total * self.unit.dx / self.unit.dt #Convert interpolated fluid velocity to physical units
        if n_lattice > 0:
            # eps_fluid = sum(eps_{i,j}) / N_lattice
            eps_fluid = eps_sum / float(n_lattice)

        # Drag force
        if 1.0 - eps_fluid > 1e-15:
            d_p = 2.0 * self.dem.gf[gid].radius
            u_slip = self.dem.gf[gid].velocity - fluid_vel

            self.dem.gf[gid].force_fluid += self._tenneti_drag(d_p, u_slip, eps_fluid)

    @ti.func
    def _tenneti_drag(self, dp: float, u_slip: Vector3, fvf: float) -> Vector3:
        """
        Calculate Tenneti drag force vector in physical units [N].
        Formula: F_d = -3π d_p μ ε_f C_d(Re_p, ε) u_slip
        """
        u_mag = tm.length(u_slip)
        Re_p = fvf * self.rho0 * dp * u_mag / self.mu

        # Compute Drag Coefficient Cd (Reuses shared logic)
        Cd = self._compute_tenneti_Cd(Re_p, fvf)

        # Drag Force Vector
        F_drag = -3.0 * tm.pi * dp * self.mu * fvf * Cd * u_slip
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
        # W(x_p) -- primary contribution
        w_primary = (self._threedelta(ti.abs(xc - ii)) *
                     self._threedelta(ti.abs(yc - jj)) *
                     self._threedelta(ti.abs(zc - kk)))

        # W(x'_p) -- mirror contributions (Eq.22)
        w_mirror = 0.0

        # -- x walls --
        if xc < support:  # near left wall (i = 0)
            xc_mir = -xc
            w_mirror += (self._threedelta(ti.abs(xc_mir - ii)) *
                         self._threedelta(ti.abs(yc - jj)) *
                         self._threedelta(ti.abs(zc - kk)))
        if xc > float(self.Nx - 1) - support:  # near right wall (i = Nx -1)
            xc_mir = 2.0 * float(self.Nx - 1) - xc
            w_mirror += (self._threedelta(ti.abs(xc_mir - ii)) *
                         self._threedelta(ti.abs(yc - jj)) *
                         self._threedelta(ti.abs(zc - kk)))

        # -- y walls --
        if yc < support:
            yc_mir = -yc
            w_mirror += (self._threedelta(ti.abs(xc - ii)) *
                         self._threedelta(ti.abs(yc_mir - jj)) *
                         self._threedelta(ti.abs(zc - kk)))
        if yc > float(self.Ny - 1) - support:
            yc_mir = 2.0 * float(self.Ny - 1) - yc
            w_mirror += (self._threedelta(ti.abs(xc - ii)) *
                         self._threedelta(ti.abs(yc_mir - jj)) *
                         self._threedelta(ti.abs(zc - kk)))

        # -- z walls --
        if zc < support:
            zc_mir = -zc
            w_mirror += (self._threedelta(ti.abs(xc - ii)) *
                         self._threedelta(ti.abs(yc - jj)) *
                         self._threedelta(ti.abs(zc_mir - kk)))
        if zc > float(self.Nz - 1) - support:
            zc_mir = 2.0 * float(self.Nz - 1) - zc
            w_mirror += (self._threedelta(ti.abs(xc - ii)) *
                         self._threedelta(ti.abs(yc - jj)) *
                         self._threedelta(ti.abs(zc_mir - kk)))

        return w_primary + w_mirror  # W_bar = W(x_p) + W(x'_p),
    # ======================================================================
    # High-level interface
    # ======================================================================
    # (To be implemented by the user / calling script)
    def initialize_complete(self):
        """Complete initialization sequence for coupling."""
        self.initialize()
        self.map_coarse_grains()
        self.map_fine_grains()
        self.compute_fine_weights()

    def update_coupling(self):
        """Update coupling fields at each time step."""
        self.map_fine_grains()
        self.map_coarse_grains()
        self.compute_fine_weights()
        self.lattice2grains()