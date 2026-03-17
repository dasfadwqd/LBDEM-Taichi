"""
Hybrid Resolved/Unresolved LBM-DEM Coupling (3D)
=================================================

Combines the Partially Saturated Cell (PSC) method for coarse particles (dp > dx)
and the Particle Equivalence IBM method (Tenneti drag) for fine particles (dp ≤ dx).

Key design decisions
--------------------
* Solid volume fraction is split into two independent fields:
      coarse_volfrac  –  contribution from resolved (large) particles
      fine_volfrac    –  contribution from unresolved (small) particles
* The collision step dispatches to one of three paths per lattice node:
      1. Pure fluid              (both volfrac == 0)
      2. Coarse-particle cell    (coarse_volfrac > 0)  → PSC bounce-back
      3. Fine-particle cell      (fine_volfrac   > 0)  → IBM weighted collision
  A cell may in principle carry both contributions (coarse + fine); in that case
  coarse-particle physics takes priority (the PSC path handles it) and the
  fine-particle contribution is folded into the background volfrac for drag only.
* Streaming step is identical to the base class – no changes required.
* Hydrodynamic forces:
      Both particle classes are handled in a single lattice2grains() kernel,
      called once per time step after stream() + compute_macro().
      Force and torque fields are zeroed at the top of lattice2grains() to
      guarantee a clean slate regardless of call order.

      Coarse particles  → accumulate hydroforce (set during collision) → convert to
                          physical force + torque via momentum-exchange.
      Fine   particles  → interpolate post-stream fluid velocity → Tenneti drag.

Step order per time step
------------------------
  1. collide()        – grain mapping, collision, accumulate hydroforce (lattice units)
  2. stream()         – propagate fpc → f  (inherited, unchanged)
  3. apply_bc()       – boundary conditions (inherited, unchanged)
  4. compute_macro()  – update ρ and u from post-stream f (inherited, unchanged)
  5. lattice2grains() – SINGLE entry point: zero forces, coarse momentum-exchange,
                        fine Tenneti drag, write to DEM force/torque fields.

References
----------
PSC method : Noble & Torczynski, Int. J. Mod. Phys. C 9 (1998) 1189-1201
Tenneti drag: Tenneti et al., Int. J. Multiphase Flow 37 (2011) 1072-1092
Equiv. IBM  : Wang et al., Chem. Eng. J. (2023) 142898

"""

import taichi as ti
import taichi.math as tm

from src.lbm3d.lbm_solver3d import BasicLattice3D
from src.lbm3d.lbmutils import CellType
from src.dem3d.demsolver import DEMSolver

# ---------------------------------------------------------------------------
# Convenience alias
# ---------------------------------------------------------------------------
Vector3 = ti.types.vector(3, float)
class HybridLattice3D(BasicLattice3D):
    """3-D LBM lattice with hybrid resolved/unresolved FSI coupling.

    Particle classification at runtime
    -----------------------------------
    For each DEM grain *id*:
        if  2 * radius  >  dx   →  **coarse** (resolved,  PSC method)
        else                    →  **fine**   (unresolved, IBM/Tenneti)

    This means the same simulation can contain a polydisperse mixture of
    particle sizes and each grain is automatically handled by the most
    appropriate coupling strategy.
    """

    # D3Q19 inverse direction table (static, used inside Taichi kernels)
    QINV_STATIC = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17)

    def __init__(
        self,
        Nx: int, Ny: int, Nz: int,
        omega: float,
        dx: float, dt: float,
        rho: float,
        demslover: DEMSolver,
    ):
        super().__init__(Nx, Ny, Nz, omega, dx, dt, rho)

        shape = (Nx, Ny, Nz)

        # ------------------------------------------------------------------
        # Fluid properties (needed for drag models)
        # ------------------------------------------------------------------
        self.rho0  = rho
        self.omega0 = omega
        self.nuLu  = (1.0 / omega - 0.5) / 3.0               # lattice kinematic viscosity
        self.nu    = self.nuLu * (dx ** 2) / dt               # physical [m²/s]
        self.mu    = rho * self.nu                             # dynamic [Pa·s]

        # ------------------------------------------------------------------
        # Shared coupling fields
        # ------------------------------------------------------------------
        # Volume fractions (split by particle class)
        self.coarse_volfrac = ti.field(float, shape=shape)    # resolved particles
        self.fine_volfrac   = ti.field(float, shape=shape)    # unresolved particles


        # Solid velocity fields (each class writes its own)
        self.coarse_velsolid = ti.Vector.field(self.D, float, shape=shape)
        self.fine_velsolid   = ti.Vector.field(self.D, float, shape=shape)


        # Weighting coefficients
        self.coarse_weight = ti.field(float, shape=shape)     # B from PSC
        self.fine_weight   = ti.field(float, shape=shape)     # beta from Tenneti

        # feq computed at solid velocity (one per class)
        self.coarse_feqsolid = ti.Vector.field(self.Q, float, shape=shape)
        self.fine_feqsolid   = ti.Vector.field(self.Q, float, shape=shape)

        # Momentum-exchange force (coarse particles only, per cell)
        self.hydroforce = ti.Vector.field(self.D, float, shape=shape)
        self.hydrotorque = ti.Vector.field(self.D, float, shape=shape)

        # Grain-id map for coarse particles (−1 → no coarse grain)
        self.coarse_id = ti.field(int, shape=shape)

        # ------------------------------------------------------------------
        # Temporary accumulators for IBM (fine particles)
        # ------------------------------------------------------------------
        self.fine_velsum    = ti.Vector.field(self.D, float, shape=shape)
        self.fine_weightsum = ti.field(float, shape=shape)

        # ------------------------------------------------------------------
        # DEM solver reference
        # ------------------------------------------------------------------
        self.dem = demslover

    # ======================================================================
    # Initialisation
    # ======================================================================

    @ti.kernel
    def initialize(self):
        """Set distribution functions to equilibrium; map grains to lattice."""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue
            self.compute_feq(i, j, k)
            for q in ti.static(range(HybridLattice3D.Q)):
                self.f[i, j, k][q] = self.feq[i, j, k][q]

    # ======================================================================
    # DEM → Lattice mapping
    # ======================================================================

    @ti.kernel
    def map_coarse_grains(self):
        """Map resolved (coarse) grains using the PSC solid-fraction method.

        A 5×5×5 sub-cell decomposition estimates the solid volume fraction ε
        for partially covered cells. Fully covered cells get ε = 1.

        Only grains with  diameter > dx  contribute here.
        """
        self.coarse_id.fill(-1)
        self.coarse_volfrac.fill(0.0)
        self.coarse_weight.fill(0.0)
        self.coarse_velsolid.fill(0.0)

        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            for gid in range(self.dem.gf.shape[0]):
                # ---- classify: skip fine particles ----
                if 2.0 * self.dem.gf[gid].radius <= self.unit.dx:
                    continue

                # Grain centre in lattice coordinates
                xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin
                      + 0.5 * self.unit.dx) / self.unit.dx
                yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin
                      + 0.5 * self.unit.dx) / self.unit.dx
                zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin
                      + 0.5 * self.unit.dx) / self.unit.dx
                # Effective radius (shrunk by half-cell so boundary cells are
                # treated as partially covered rather than fully solid)
                r   = (self.dem.gf[gid].radius - 0.5 * self.unit.dx) / self.unit.dx

                dist = ti.sqrt((xc - i)**2 + (yc - j)**2 + (zc - k)**2)

                # ---- fully outside ----
                if dist >= r + 0.5 * ti.sqrt(3.0):
                    continue

                # ---- fully inside ----
                if dist <= r - 0.5 * ti.sqrt(3.0):
                    self.coarse_id[i, j, k]      = gid
                    self.coarse_volfrac[i, j, k] = 1.0
                    self.set_coarse_weight(i, j, k)
                    self.set_coarse_velsolid(i, j, k, gid, xc, yc, zc)
                    break  # one grain dominates per cell

                # ---- partially covered: 5³ sub-cell integration ----
                cnt = 0
                for si in range(5):
                    for sj in range(5):
                        for sk in range(5):
                            d2 = ti.sqrt(
                                (xc - (i - 0.4 + 0.2 * si))**2 +
                                (yc - (j - 0.4 + 0.2 * sj))**2 +
                                (zc - (k - 0.4 + 0.2 * sk))**2
                            )
                            if d2 < r:
                                cnt += 1
                eps = cnt / 125.0

                # keep the grain with the largest coverage per cell
                if eps > self.coarse_volfrac[i, j, k]:
                    self.coarse_id[i, j, k]      = gid
                    self.coarse_volfrac[i, j, k] = eps
                    self.set_coarse_weight(i, j, k)
                    self.set_coarse_velsolid(i, j, k, gid, xc, yc, zc)

    @ti.func
    def set_coarse_weight(self, i: int, j: int, k: int):
        """PSC weighting coefficient  B = ε(τ-½) / [(1-ε)+(τ-½)]."""
        eps   = self.coarse_volfrac[i, j, k]
        tau_m = 1.0 / self.omega[i, j, k] - 0.5
        self.coarse_weight[i, j, k] = (eps * tau_m) / ((1.0 - eps) + tau_m)

    @ti.func
    def set_coarse_velsolid(
        self, i: int, j: int, k: int,
        gid: int, xc: float, yc: float, zc: float
    ):
        """Solid velocity = translational + ω×r  (converted to lattice units)."""
        r_vec         = Vector3(i, j, k) - Vector3(xc, yc, zc)
        omega_cross_r = tm.cross(self.dem.gf[gid].omega, r_vec * self.unit.dx)
        self.coarse_velsolid[i, j, k] = (
            (self.dem.gf[gid].velocity + omega_cross_r) * self.unit.dt / self.unit.dx
        )

    # ------------------------------------------------------------------

    @ti.kernel
    def map_fine_grains(self):
        """Map unresolved (fine) grains using a regularised IBM kernel.

        Uses the 3-point delta function (threedelta) with support radius 1.5 dx.
        Weight redistribution corrects for nodes near domain boundaries.

        Only grains with  diameter ≤ dx  contribute here.
        """
        self.fine_volfrac.fill(0.0)
        self.fine_velsum.fill(0.0)
        self.fine_weightsum.fill(0.0)

        V_lattice = self.unit.dx ** 3

        for gid in range(self.dem.gf.shape[0]):
            # ---- classify: skip coarse particles ----
            if 2.0 * self.dem.gf[gid].radius > self.unit.dx:
                continue

            xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin
                  + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin
                  + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin
                  + 0.5 * self.unit.dx) / self.unit.dx
            V_grain    = 4.0 / 3.0 * tm.pi * self.dem.gf[gid].radius ** 3
            vel_lattice = self.dem.gf[gid].velocity * self.unit.dt / self.unit.dx
            support_radius = 1.5

            i_min = ti.max(0, ti.cast(xc - support_radius, ti.i32))
            i_max = ti.min(self.Nx, ti.cast(xc + support_radius + 1, ti.i32))
            j_min = ti.max(0, ti.cast(yc - support_radius, ti.i32))
            j_max = ti.min(self.Ny, ti.cast(yc + support_radius + 1, ti.i32))
            k_min = ti.max(0, ti.cast(zc - support_radius, ti.i32))
            k_max = ti.min(self.Nz, ti.cast(zc + support_radius + 1, ti.i32))

            # First pass – total vs valid weight (boundary correction)
            total_w = 0.0
            valid_w = 0.0
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    for k in range(k_min, k_max):
                        dist_x = ti.abs(xc - i)
                        dist_y = ti.abs(yc - j)
                        dist_z = ti.abs(zc - k)
                        weight_x = self._threedelta(dist_x)
                        weight_y = self._threedelta(dist_y)
                        weight_z = self._threedelta(dist_z)
                        w    = weight_x * weight_y * weight_z
                        if w <= 0.0:
                            continue
                        total_w += w
                        if not (self.CT[i, j, k] &
                                (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP)):
                            valid_w += w

            if valid_w < 1e-12:
                continue

            corr = total_w / valid_w          # redistribute excluded-node weight

            # Second pass – accumulate into lattice fields
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    for k in range(k_min, k_max):
                        if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD |
                                               CellType.FREE_SLIP):
                            continue
                        dist_x = ti.abs(xc - i)
                        dist_y = ti.abs(yc - j)
                        dist_z = ti.abs(zc - k)
                        weight_x = self._threedelta(dist_x)
                        weight_y = self._threedelta(dist_y)
                        weight_z = self._threedelta(dist_z)
                        w = weight_x * weight_y * weight_z
                        if w <= 0.0:
                            continue
                        w_c = w * corr
                        ti.atomic_add(self.fine_volfrac[i, j, k],
                                      w_c * V_grain / V_lattice)
                        ti.atomic_add(self.fine_velsum[i, j, k],  w_c * vel_lattice)
                        ti.atomic_add(self.fine_weightsum[i, j, k], w_c)

        # Normalise velocity and clamp volfrac
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.fine_volfrac[i, j, k] >= 1.0:
                self.fine_volfrac[i, j, k] = 0.9999
                print("Warning: fine particles volfrac[{}, {}, {}] >= 1.0".format(i, j, k))
            if self.fine_weightsum[i, j, k] > 1e-10:
                self.fine_velsolid[i, j, k] =  self.fine_velsum[i, j, k] / self.fine_weightsum[i, j, k]

    # ======================================================================
    # Collision step
    # ======================================================================

    @ti.kernel
    def collide(self):
        """Hybrid collision step.

        Per-cell dispatch:
            coarse_volfrac > 0  →  PSC bounce-back collision
            fine_volfrac   > 0  →  IBM weighted collision (Tenneti)
            otherwise           →  standard BGK
        Coarse takes priority when a cell has contributions from both classes.
        """
        # reset hydrodynamic force and torque to zero
        self.hydroforce.fill(0.0)
        self.hydrotorque.fill(0.0)

        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            self.compute_feq(i, j, k)

            if self.coarse_volfrac[i, j, k] > 0.0 >= self.fine_volfrac[i, j, k]:
                # --- Resolved particle: PSC momentum-exchange ---
                self.collide_coarse(i, j, k)

            elif self.fine_volfrac[i, j, k] > 0.0 >= self.coarse_volfrac[i, j, k]:
                # --- Unresolved particle: IBM weighted collision ---
                self.collide_fine(i, j, k)

            elif self.fine_volfrac[i, j, k] > 0.0 and self.coarse_volfrac[i, j, k] > 0.0:
                self.collide_hybrid(i, j, k)
            else:
                # --- Pure fluid ---
                self.collide_fluid(i, j, k)



    # ------------------------------------------------------------------
    # Collision sub-routines
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
        """PSC collision for resolved (coarse) particles.

        f_post = f + B·Ω_s + (1-B)·Ω_f

        Ω_s = f[q̄] - feq[q̄] + feq_solid[q] - f[q]   (bounce-back with solid feq)
        Ω_f = -ω·(f[q] - feq[q])                       (BGK)

        Hydrodynamic force accumulated via momentum exchange.
        """
        self.compute_feq_solid_coarse(i, j, k)
        B = self.coarse_weight[i, j, k]

        for q in range(HybridLattice3D.Q):
            q_inv = HybridLattice3D.QINV_STATIC[q]
            Omega_s = (
                self.f[i, j, k][q_inv]
                - self.feq[i, j, k][q_inv]
                + self.coarse_feqsolid[i, j, k][q]
                - self.f[i, j, k][q]
            )
            Omega_f = -self.omega[i, j, k] * (self.f[i, j, k][q] - self.feq[i, j, k][q])
            self.fpc[i, j, k][q] = (
                self.f[i, j, k][q] + B * Omega_s + (1.0 - B) * Omega_f
            )
            # Momentum exchange: F = -B · Ω_s · c  (lattice units)
            self.hydroforce[i, j, k] -= B * Omega_s * HybridLattice3D.c[q]

    @ti.func
    def collide_fine(self, i: int, j: int, k: int):
        """IBM-style weighted collision for unresolved (fine) particles.

        Uses the same PSC-like operator but the weight W_d comes from the
        Tenneti drag model rather than from a geometric solid fraction.

        f_post = f +β·Ω_s + (1-β)·Ω_f

        No per-cell momentum exchange is stored here; forces on fine
        particles are computed directly in lattice2grains() via the drag model.
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
            self.fpc[i, j, k][q] = (
                self.f[i, j, k][q] + beta * Omega_s + (1.0 - beta) * Omega_f
            )

    @ti.func
    def collide_hybrid(self , i:int, j: int, k:int):
        """
        f_post = f +β·Ω_s1 + B·Ω_s2 (1-β - B)·Ω_f
        """
        self.compute_feq_solid_fine(i, j, k)
        self.compute_feq_solid_coarse(i, j, k)

        B = self.coarse_weight[i, j, k]
        beta = self.fine_weight[i, j, k]

        for q in range(HybridLattice3D.Q):
            q_inv = HybridLattice3D.QINV_STATIC[q]
            Omega_s1 = (
                    self.f[i, j, k][q_inv]
                    - self.feq[i, j, k][q_inv]
                    + self.fine_feqsolid[i, j, k][q]
                    - self.f[i, j, k][q]
            )
            Omega_s2 = (
                    self.f[i, j, k][q_inv]
                    - self.feq[i, j, k][q_inv]
                    + self.coarse_feqsolid[i, j, k][q]
                    - self.f[i, j, k][q]
            )
            Omega_f = -self.omega[i, j, k] * (self.f[i, j, k][q] - self.feq[i, j, k][q])
            if B + beta > 1.0:
                beta = 1.0 - B
            self.fpc[i, j, k][q] = (
                    self.f[i, j, k][q] + beta * Omega_s1 +  B * Omega_s2 + (1.0 - B - beta) * Omega_f
            )
            # Momentum exchange: F = -B · Ω_s · c  (lattice units)
            self.hydroforce[i, j, k] -= B * Omega_s1 * HybridLattice3D.c[q]


    # ------------------------------------------------------------------
    # feq helpers
    # ------------------------------------------------------------------

    @ti.func
    def compute_feq_solid_coarse(self, i: int, j: int, k: int):
        """feq computed at the coarse-grain solid velocity."""
        u  = self.coarse_velsolid[i, j, k]
        uv = tm.dot(u, u)
        for q in range(HybridLattice3D.Q):
            cu = tm.dot(HybridLattice3D.c[q], u)
            self.coarse_feqsolid[i, j, k][q] = (
                HybridLattice3D.w[q] * self.rho[i, j, k]
                * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uv)
            )

    @ti.func
    def compute_feq_solid_fine(self, i: int, j: int, k: int):
        """feq computed at the fine-grain solid velocity."""
        u  = self.fine_velsolid[i, j, k]
        uv = tm.dot(u, u)
        for q in ti.static(range(HybridLattice3D.Q)):
            cu = tm.dot(HybridLattice3D.c[q], u)
            self.fine_feqsolid[i, j, k][q] = (
                HybridLattice3D.w[q] * self.rho[i, j, k]
                * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uv)
            )

    # ======================================================================
    # Weight coefficient for fine particles (Tenneti model)
    # ======================================================================

    @ti.kernel
    def _compute_fine_weights(self):
        """Compute per-cell IBM weight coefficient W_d using the Tenneti drag model.

        W_d = 3π d_p ν_L (1-ε) C_d   (dimensionless, lattice units)

        where d_p is the equivalent sphere diameter derived from the local
        volume fraction and lattice cell volume.
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
        """Tenneti drag coefficient in lattice-unit dimensionless form.

        Returns W_d = 3π d_p^L ν_L (1-ε) C_d   (≤ 1 expected for stability).
        """
        u_mag = tm.length(u_slip)
        Re_p  = (1.0 - svf) * self.rho0 * dp * u_mag / self.mu

        C_d = 0.0
        if (1.0 - svf) > 1e-9:
            Cd0   = 1.0 + 0.15 * tm.pow(Re_p, 0.687)
            A_eps = (5.81 * svf / (1.0 - svf)**3
                     + 0.48 * tm.pow(svf, 1.0/3.0) / (1.0 - svf)**4)
            svf3  = svf ** 3
            B_eps = svf3 * Re_p * (0.95 + 0.61 * svf3 / (1.0 - svf)**2)
            C_d   = (1.0 - svf) * (Cd0 / (1.0 - svf)**3 + A_eps + B_eps)

        dp_L = dp / self.unit.dx          # diameter in lattice units
        Wd   = 3.0 * tm.pi * dp_L * self.nuLu * (1.0 - svf) * C_d
        return Wd

    # ======================================================================
    # Lattice → Grains force transfer
    # ======================================================================

    # ======================================================================
    # Unified force transfer: Lattice → Grains
    # ======================================================================

    @ti.kernel
    def lattice2grains(self):
        """Transfer hydrodynamic forces from the lattice to all DEM particles.

        This is the **single** public entry point for fluid→particle force
        coupling. It handles both particle classes in one kernel so that:

        * Force and torque fields are zeroed exactly once, at the top, before
          any accumulation occurs. This prevents stale values from previous
          time steps from corrupting the result regardless of external call order.
        * Coarse-particle (PSC) and fine-particle (Tenneti) paths execute in
          the same kernel launch, avoiding inter-kernel race conditions.

        Must be called AFTER stream() + compute_macro() so that self.vel
        reflects the post-stream macroscopic fluid velocity.

        Call order within a time step
        ------------------------------
          collide()        ← fills self.hydroforce (lattice units)
          stream()         ← propagates fpc → f
          apply_bc()
          compute_macro()  ← updates self.vel (physical velocity used below)
          lattice2grains() ← THIS function

        Coarse particles  (dp > dx) — PSC momentum exchange
        ----------------------------------------------------
          F_phys = Σ_cells  hydroforce[cell] · ρ dx⁴ / dt²
          T_phys = Σ_cells  r_vec × F_phys
          where r_vec = (x_grain − x_cell) in physical units.

        Fine particles  (dp ≤ dx) — Tenneti drag model
        ------------------------------------------------
          Interpolate u_fluid and ε_p at grain position via IBM kernel,
          then compute F_drag = −3π dp μ (1−ε) C_d(Re_p, ε) u_slip.
          No torque term (unresolved particles are point-like).
        """
        # ------------------------------------------------------------------
        # Step 0 – Zero all fluid forces and torques on every grain.
        #          Done here (not in collide) so this function is self-contained
        #          and safe to call independently of collide().
        # ------------------------------------------------------------------
        self.dem.gf.force_fluid.fill(0.0)
        self.dem.gf.moment_fluid.fill(0.0)

        # ------------------------------------------------------------------
        # Step 1 – Coarse particles: PSC momentum exchange
        # ------------------------------------------------------------------
        for gid in range(self.dem.gf.shape[0]):
            # Skip fine particles in this pass
            if 2.0 * self.dem.gf[gid].radius <= self.unit.dx:
                continue

            xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin
                  + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin
                  + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin
                  + 0.5 * self.unit.dx) / self.unit.dx
            r  = self.dem.gf[gid].radius / self.unit.dx

            x0 = ti.max(0,       int(xc - r))
            x1 = ti.min(self.Nx, int(xc + r + 2))
            y0 = ti.max(0,       int(yc - r))
            y1 = ti.min(self.Ny, int(yc + r + 2))
            z0 = ti.max(0,       int(zc - r))
            z1 = ti.min(self.Nz, int(zc + r + 2))

            for i in range(x0, x1):
                for j in range(y0, y1):
                    for k in range(z0, z1):
                        if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD |
                                               CellType.FREE_SLIP):
                            continue
                        if self.coarse_volfrac[i, j, k] > 0.0 and self.coarse_id[i, j, k] == gid:
                            # Convert lattice-unit momentum exchange to physical force
                            Ff = (self.hydroforce[i, j, k]
                                  * self.unit.rho * self.unit.dx ** 4 / self.unit.dt ** 2)
                            self.dem.gf[gid].force_fluid += Ff
                            # Torque about grain centre (r_vec: grain → cell)
                            r_vec = (Vector3(xc, yc, zc) - Vector3(i, j, k)) * self.unit.dx
                            self.dem.gf[gid].moment_fluid += tm.cross(r_vec, Ff)

        # ------------------------------------------------------------------
        # Step 2 – Fine particles: Tenneti drag via IBM interpolation
        # ------------------------------------------------------------------
        for gid in range(self.dem.gf.shape[0]):
            # Skip coarse particles in this pass
            if 2.0 * self.dem.gf[gid].radius > self.unit.dx:
                continue

            xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin
                  + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin
                  + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin
                  + 0.5 * self.unit.dx) / self.unit.dx


            x0 = ti.max(0,       int(xc - 2))
            x1 = ti.min(self.Nx, int(xc + 2))
            y0 = ti.max(0,       int(yc - 2))
            y1 = ti.min(self.Ny, int(yc + 2))
            z0 = ti.max(0,       int(zc - 2))
            z1 = ti.min(self.Nz, int(zc + 2))

            # IBM interpolation: accumulate weighted fluid velocity and ε at grain position
            u_fluid_particle = Vector3(0.0, 0.0, 0.0)
            volfrac_particle = 0.0

            for i in range(x0, x1):
                for j in range(y0, y1):
                    for k in range(z0, z1):
                        if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD |
                                               CellType.FREE_SLIP):
                            continue
                        dist_x = ti.abs(xc - i)
                        dist_y = ti.abs(yc - j)
                        dist_z = ti.abs(zc - k)
                        weight_x = self._threedelta(dist_x)
                        weight_y = self._threedelta(dist_y)
                        weight_z = self._threedelta(dist_z)
                        w    = weight_x * weight_y * weight_z
                        # self.vel is in lattice units; convert to physical below
                        u_fluid_particle += self.vel[i, j, k] * w
                        volfrac_particle   += self.fine_volfrac[i, j, k] * w

            if volfrac_particle  > 0.0:
                # Convert interpolated fluid velocity from lattice to physical units
                u_fluid = u_fluid_particle   * self.unit.dx / self.unit.dt
                d_p     = 2.0 * self.dem.gf[gid].radius
                u_slip  = self.dem.gf[gid].velocity - u_fluid
                self.dem.gf[gid].force_fluid += self._tenneti_drag(d_p, u_slip, volfrac_particle)

    @ti.func
    def _tenneti_drag(self, dp: float, u_slip: Vector3, svf: float) -> Vector3:
        """Tenneti drag force vector in physical units [N].

        F_d = -3π d_p μ (1-ε) C_d(Re_p, ε) u_slip
        """
        u_mag = tm.length(u_slip)
        Re_p  = (1.0 - svf) * self.rho0 * dp * u_mag / self.mu

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
        F_drag = - 3.0 * tm.pi * dp * self.mu * (1.0 - svf) * C_d * u_slip

        return F_drag


    # ======================================================================
    # IBM kernel (3-point regularised delta function)
    # ======================================================================

    @ti.func
    def _threedelta(self, r: float) -> float:
        """Peskin 3-point delta function kernel (support: |r| ≤ 1.5)."""
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

