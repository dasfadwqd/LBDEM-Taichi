"""
Source-term-based unresolved LBM-DEM coupling.

This module implements a D3Q19 volume-averaged formulation based on Xu et al.
(2026). BGK and MRT collision operators are selectable. Porosity variation enters the
lattice Boltzmann equation through a mass source, while the reaction to the
particle drag enters through a momentum source.  The particle drag is the
Tenneti correlation already used by :mod:`src.unlbdem.eqlattice`.

Particle volume and the particle-to-fluid reaction force are distributed with
the same tensor-product delta kernel (three 1-D threedelta weights multiplied
along each axis, then normalized per particle). Periodic directions wrap the
kernel stencil; non-periodic domain faces mirror its truncated lobe back onto
the interior fluid nodes. Fluid variables at a particle center are recovered
by the same weighted sum.

The time-step order is::

    _swap_history()   # save eps/gfield/Ffield of the previous step
    grains2lattice()  # map particle volume -> eps_s/eps_f
    lattice2grains()  # interpolate fluid -> drag -> distribute reaction to hydroforce
    assemble_source() # Sq/Sm -> directional sources gfield/Ffield
    collide()         # selectable BGK/MRT relaxation + explicit sources
    stream()          # inherited standard streaming and macro update
"""

import numpy as np
import taichi as ti
import taichi.math as tm

from src.dem3d.demsolver import DEMSolver
from src.lbm3d.lbm_solver3d import BasicLattice3D
from src.lbm3d.lbmutils import CellType


Vector3 = ti.types.vector(3, float)

# 排除在映射 / 源项 / 碰撞之外的边界格点类型。
BOUNDARY_MASK = (
    CellType.OBSTACLE
    | CellType.VEL_LADD
    | CellType.FREE_SLIP
    | CellType.VEL_INLET_LADD
)

# 颗粒—格点耦合不写入任何外边界节点。与 BOUNDARY_MASK 不同，压力、
# Zou/He 及 Skordos 节点虽然参与边界重构，但不应承接颗粒体积或反作用力；
# 非周期方向越过这些端面时，核权重镜像折回相邻内部流体节点。
MAPPING_BOUNDARY_MASK = (
    BOUNDARY_MASK
    | CellType.VEL_ZOUHE
    | CellType.VEL_EXIT
    | CellType.Pre_ZOUHE
    | CellType.VEL_SKORDOS
    | CellType.PRE_SKORDOS
)

# Wet-node velocity/pressure boundaries use their prescribed base relaxation
# frequency.  Only interior fluid nodes receive the Smagorinsky correction.
WET_NODE_BOUNDARY_MASK = (
    CellType.VEL_ZOUHE
    | CellType.VEL_EXIT
    | CellType.Pre_ZOUHE
    | CellType.VEL_SKORDOS
    | CellType.PRE_SKORDOS
)

SOLID_WALL_MASK = CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP


@ti.data_oriented
class SourceTermLattice3D(BasicLattice3D):
    """D3Q19 unresolved LBM-DEM solver with VANSE source terms.

    The recovered target equations are the Model-A volume-averaged equations.
    In lattice units, the auxiliary source terms are

    ``Sq = -rho / eps_f * (d(eps_f)/dt + u . grad(eps_f))``

    ``Sm = Sq * u + (F_pf + F) / eps_f``

    where ``F_pf`` is the reaction-force density exerted by particles on the
    fluid and ``F`` is an optional body-force density. Macroscopic density and
    velocity retain the standard definitions inherited from
    :class:`BasicLattice3D`.
    """

    MIN_FLUID_FRACTION = 1.0e-6
    WALE_CONSTANT = 0.325
    VON_KARMAN_CONSTANT = 0.41

    def __init__(self, Nx: int, Ny: int, Nz: int, omega: float,
                 dx: float, dt: float, rho: float, demsolver: DEMSolver,
                 periodic=(False, False, False), collision_model="bgk",
                 turbulence_model="none"):
        super().__init__(Nx, Ny, Nz, omega, dx, dt, rho)

        collision_model = collision_model.lower()
        turbulence_model = turbulence_model.lower()
        if collision_model not in ("bgk", "mrt"):
            raise ValueError("collision_model must be 'bgk' or 'mrt'")
        if turbulence_model not in ("none", "smagorinsky", "wale"):
            raise ValueError(
                "turbulence_model must be 'none', 'smagorinsky' or 'wale'"
            )
        self.collision_model = collision_model
        self.turbulence_model = turbulence_model

        self.dem = demsolver
        self.periodic_x = bool(periodic[0])
        self.periodic_y = bool(periodic[1])
        self.periodic_z = bool(periodic[2])
        self.rho0 = rho
        self.nuLu = (1.0 / omega - 0.5) / 3.0
        self.nu = self.nuLu * dx * dx / dt
        self.mu = rho * self.nu

        # Non-orthogonal D3Q19 polynomial moment basis. The first four rows
        # are the conserved density and momentum moments; rows 5--9 are the
        # five deviatoric/shear-stress moments controlled by the local
        # molecular+SGS viscosity.
        mrt_matrix = self._build_mrt_matrix()
        self.mrt_matrix = ti.field(float, shape=(self.Q, self.Q))
        self.mrt_inverse = ti.field(float, shape=(self.Q, self.Q))
        self.mrt_relaxation = ti.field(float, shape=(self.Q,))
        self.mrt_matrix.from_numpy(mrt_matrix)
        self.mrt_inverse.from_numpy(np.linalg.inv(mrt_matrix))
        self.mrt_relaxation.from_numpy(np.array([
            0.0, 0.0, 0.0, 0.0,  # rho, jx, jy, jz
            1.0,                  # kinetic-energy/bulk mode
            0.0, 0.0, 0.0, 0.0, 0.0,  # local shear rate, set in collide
            1.2, 1.2, 1.2, 1.2, 1.2, 1.2,  # third-order modes
            1.4, 1.4, 1.4,       # fourth-order modes
        ], dtype=np.float64))

        # Eulerian 相场
        self.volfrac = ti.field(float, shape=(Nx, Ny, Nz))          # 固含率 eps_s
        self.fluid_fraction = ti.field(float, shape=(Nx, Ny, Nz))   # 孔隙率 eps_f
        self.prev_fluid_fraction = ti.field(float, shape=(Nx, Ny, Nz))

        # 颗粒反作用力密度（Eq.48），格子单位。
        self.hydroforce = ti.Vector.field(self.D, float, shape=(Nx, Ny, Nz))
        # 可选非颗粒体积力密度（Eq.14），格子单位。
        self.body_force = ti.Vector.field(self.D, float, shape=(Nx, Ny, Nz))

        # 方向源分布及其上一时步值（Eq.22/23）。
        self.gfield = ti.Vector.field(self.Q, float, shape=(Nx, Ny, Nz))
        self.gfield_prev = ti.Vector.field(self.Q, float, shape=(Nx, Ny, Nz))
        self.Ffield = ti.Vector.field(self.Q, float, shape=(Nx, Ny, Nz))
        self.Ffield_prev = ti.Vector.field(self.Q, float, shape=(Nx, Ny, Nz))

        self.fluid_fraction.fill(1.0)
        self.prev_fluid_fraction.fill(1.0)
        self.body_force.fill(0.0)

    @staticmethod
    def _build_mrt_matrix():
        """Return a full-rank polynomial moment basis for this D3Q19 order."""
        c = np.array([
            [0, 0, 0], [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
            [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
            [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
        ], dtype=np.float64)
        cx, cy, cz = c.T
        return np.array([
            np.ones(c.shape[0]),
            cx, cy, cz,
            cx * cx + cy * cy + cz * cz,
            cx * cx - cy * cy,
            cy * cy - cz * cz,
            cx * cy, cx * cz, cy * cz,
            cx * cx * cy, cx * cx * cz,
            cx * cy * cy, cy * cy * cz,
            cx * cz * cz, cy * cz * cz,
            cx * cx * cy * cy,
            cx * cx * cz * cz,
            cy * cy * cz * cz,
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # 权重函数（张量积核）
    # ------------------------------------------------------------------

    @ti.func
    def threedelta(self, r: float) -> float:
        """一维三点 delta 权重（Roma 3-point），支撑半径 1.5 lu。"""
        a = 0.0
        if r < 0.5:
            x = -3.0 * r ** 2 + 1.0
            a = (1.0 + ti.sqrt(x)) / 3.0
        elif r <= 1.5:
            x = -3.0 * (1.0 - r) ** 2 + 1.0
            a = (5.0 - 3.0 * r - ti.sqrt(x)) / 6.0
        return a

    @ti.func
    def _kernel_weight(self, xc: float, yc: float, zc: float,
                       ii: int, jj: int, kk: int) -> float:
        """张量距离权重：三方向一维 delta 权重的乘积 w = φ(dx)·φ(dy)·φ(dz)。"""
        wx = self.threedelta(ti.abs(xc - ii))
        wy = self.threedelta(ti.abs(yc - jj))
        wz = self.threedelta(ti.abs(zc - kk))
        return wx * wy * wz

    @ti.func
    def _is_mapping_cell(self, i: int, j: int, k: int):
        return (self.CT[i, j, k] & MAPPING_BOUNDARY_MASK) == 0

    @ti.func
    def _is_source_cell(self, i: int, j: int, k: int):
        return (self.CT[i, j, k] & BOUNDARY_MASK) == 0

    @ti.func
    def _particle_lattice_coordinates(self, pid: int):
        """Particle position in lattice-index coordinates.

        Non-periodic directions retain one boundary node outside the physical
        domain (first fluid centre at index 1).  Periodic directions contain
        physical cells only (first centre at index 0).
        """
        x_shift = 0.5
        y_shift = 0.5
        z_shift = 0.5
        if ti.static(self.periodic_x):
            x_shift = -0.5
        if ti.static(self.periodic_y):
            y_shift = -0.5
        if ti.static(self.periodic_z):
            z_shift = -0.5
        xc = ((self.dem.gf[pid].position[0] - self.dem.config.domain.xmin)
              / self.unit.dx + x_shift)
        yc = ((self.dem.gf[pid].position[1] - self.dem.config.domain.ymin)
              / self.unit.dx + y_shift)
        zc = ((self.dem.gf[pid].position[2] - self.dem.config.domain.zmin)
              / self.unit.dx + z_shift)
        return xc, yc, zc

    @ti.func
    def _mapping_index_x(self, raw_i: int):
        """周期方向折回；非周期端面把核叶片镜像到内部节点。"""
        i = raw_i
        valid = 1
        if ti.static(self.periodic_x):
            i = (raw_i + self.Nx) % self.Nx
        else:
            # 非周期域的端点 0/N-1 为边界节点，物理边界面位于
            # 0.5 和 N-1.5；例如 raw_i=0 镜像到 1，raw_i=-1 到 2。
            if raw_i <= 0:
                i = 1 - raw_i
            elif raw_i >= self.Nx - 1:
                i = 2 * self.Nx - 3 - raw_i
            if self.Nx <= 2 or i < 1 or i > self.Nx - 2:
                i = ti.max(0, ti.min(self.Nx - 1, i))
                valid = 0
        return i, valid

    @ti.func
    def _mapping_index_y(self, raw_j: int):
        """周期方向折回；非周期端面把核叶片镜像到内部节点。"""
        j = raw_j
        valid = 1
        if ti.static(self.periodic_y):
            j = (raw_j + self.Ny) % self.Ny
        else:
            if raw_j <= 0:
                j = 1 - raw_j
            elif raw_j >= self.Ny - 1:
                j = 2 * self.Ny - 3 - raw_j
            if self.Ny <= 2 or j < 1 or j > self.Ny - 2:
                j = ti.max(0, ti.min(self.Ny - 1, j))
                valid = 0
        return j, valid

    @ti.func
    def _mapping_index_z(self, raw_k: int):
        """周期方向折回；非周期端面把核叶片镜像到内部节点。"""
        k = raw_k
        valid = 1
        if ti.static(self.periodic_z):
            k = (raw_k + self.Nz) % self.Nz
        else:
            if raw_k <= 0:
                k = 1 - raw_k
            elif raw_k >= self.Nz - 1:
                k = 2 * self.Nz - 3 - raw_k
            if self.Nz <= 2 or k < 1 or k > self.Nz - 2:
                k = ti.max(0, ti.min(self.Nz - 1, k))
                valid = 0
        return k, valid

    @ti.func
    def _source_index_x(self, raw_i: int):
        """源项索引：周期折回，非周期越界无效（不做镜像）。"""
        i = raw_i
        valid = 1
        if ti.static(self.periodic_x):
            i = (raw_i + self.Nx) % self.Nx
        elif raw_i < 0 or raw_i >= self.Nx:
            i = ti.max(0, ti.min(self.Nx - 1, raw_i))
            valid = 0
        return i, valid

    @ti.func
    def _source_index_y(self, raw_j: int):
        """源项索引：周期折回，非周期越界无效（不做镜像）。"""
        j = raw_j
        valid = 1
        if ti.static(self.periodic_y):
            j = (raw_j + self.Ny) % self.Ny
        elif raw_j < 0 or raw_j >= self.Ny:
            j = ti.max(0, ti.min(self.Ny - 1, raw_j))
            valid = 0
        return j, valid

    @ti.func
    def _source_index_z(self, raw_k: int):
        """源项索引：周期折回，非周期越界无效（不做镜像）。"""
        k = raw_k
        valid = 1
        if ti.static(self.periodic_z):
            k = (raw_k + self.Nz) % self.Nz
        elif raw_k < 0 or raw_k >= self.Nz:
            k = ti.max(0, ti.min(self.Nz - 1, raw_k))
            valid = 0
        return k, valid

    # ------------------------------------------------------------------
    # Kernel A: 颗粒 -> 格点映射（张量核 + 归一化）
    # ------------------------------------------------------------------

    @ti.kernel
    def grains2lattice(self):
        """把颗粒体积映射到格点。

        对每个颗粒先求和得到归一化分母 ``denom = Σ w``，再用
        ``w_norm = w / denom`` 分配体积：``eps_ij = w_norm·V_part/V_cell``。
        周期方向折回原域，非周期端面的核叶片镜像到内部流体节点。映射完
        固含率 ``eps_s`` 后取 ``eps_f = 1 - eps_s``。
        """
        self.volfrac.fill(0.0)

        V_lattice = self.unit.dx ** 3
        support = 1.5

        for pid in range(self.dem.gf.shape[0]):
            xc, yc, zc = self._particle_lattice_coordinates(pid)

            V_part = 4.0 / 3.0 * tm.pi * self.dem.gf[pid].radius ** 3
            i_min = ti.cast(ti.floor(xc - support), ti.i32)
            i_max = ti.cast(ti.floor(xc + support), ti.i32) + 1
            j_min = ti.cast(ti.floor(yc - support), ti.i32)
            j_max = ti.cast(ti.floor(yc + support), ti.i32) + 1
            k_min = ti.cast(ti.floor(zc - support), ti.i32)
            k_max = ti.cast(ti.floor(zc + support), ti.i32) + 1

            # 第一遍：归一化分母 Σ w。
            denom = 0.0
            for raw_i in range(i_min, i_max):
                for raw_j in range(j_min, j_max):
                    for raw_k in range(k_min, k_max):
                        ii, valid_i = self._mapping_index_x(raw_i)
                        jj, valid_j = self._mapping_index_y(raw_j)
                        kk, valid_k = self._mapping_index_z(raw_k)
                        if not (valid_i and valid_j and valid_k):
                            continue
                        if not self._is_mapping_cell(ii, jj, kk):
                            continue
                        w = self._kernel_weight(xc, yc, zc, raw_i, raw_j, raw_k)
                        if w <= 0.0:
                            continue
                        denom += w

            if denom < 1e-30:
                continue

            # 第二遍：用归一化权重分配体积。
            for raw_i in range(i_min, i_max):
                for raw_j in range(j_min, j_max):
                    for raw_k in range(k_min, k_max):
                        ii, valid_i = self._mapping_index_x(raw_i)
                        jj, valid_j = self._mapping_index_y(raw_j)
                        kk, valid_k = self._mapping_index_z(raw_k)
                        if not (valid_i and valid_j and valid_k):
                            continue
                        if not self._is_mapping_cell(ii, jj, kk):
                            continue
                        w = self._kernel_weight(xc, yc, zc, raw_i, raw_j, raw_k)
                        if w <= 0.0:
                            continue

                        w_norm = w / denom
                        eps_ij = w_norm * V_part / V_lattice

                        ti.atomic_add(self.volfrac[ii, jj, kk], eps_ij)

        # 由固含率补出孔隙率。
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.volfrac[i, j, k] >= 1.0:
                print("Warning: volfrac[{}, {}, {}] >= 1.0".format(i, j, k))
                self.volfrac[i, j, k] = 0.99
            self.fluid_fraction[i, j, k] = 1.0 - self.volfrac[i, j, k]

    # ------------------------------------------------------------------
    # Kernel B: 格点 -> 颗粒（张量核插值 + Tenneti 曳力 + 反作用力）
    # ------------------------------------------------------------------

    @ti.kernel
    def lattice2grains(self):
        """插值流体状态到颗粒，求曳力，并把反作用力分配到 hydroforce 场（Eq.48）。

        流体速度和局部孔隙率均按核权重归一插值。反作用力用与体积映射
        一致的周期/镜像归一化权重 ``w_norm`` 反向分配到 ``hydroforce``，
        保证动量守恒。
        """
        self.dem.gf.force_fluid.fill(0.0)
        self.hydroforce.fill(0.0)

        force_scale = self.unit.dt ** 2 / (self.unit.rho * self.unit.dx ** 4)
        support = 1.5

        for pid in range(self.dem.gf.shape[0]):
            xc, yc, zc = self._particle_lattice_coordinates(pid)

            i_min = ti.cast(ti.floor(xc - support), ti.i32)
            i_max = ti.cast(ti.floor(xc + support), ti.i32) + 1
            j_min = ti.cast(ti.floor(yc - support), ti.i32)
            j_max = ti.cast(ti.floor(yc + support), ti.i32) + 1
            k_min = ti.cast(ti.floor(zc - support), ti.i32)
            k_max = ti.cast(ti.floor(zc + support), ti.i32) + 1

            # 第一遍：加权累加流体速度与孔隙率。
            vel_wsum = Vector3(0.0, 0.0, 0.0)
            eps_wsum = 0.0
            w_total = 0.0
            for raw_i in range(i_min, i_max):
                for raw_j in range(j_min, j_max):
                    for raw_k in range(k_min, k_max):
                        ii, valid_i = self._mapping_index_x(raw_i)
                        jj, valid_j = self._mapping_index_y(raw_j)
                        kk, valid_k = self._mapping_index_z(raw_k)
                        if not (valid_i and valid_j and valid_k):
                            continue
                        if not self._is_mapping_cell(ii, jj, kk):
                            continue
                        w = self._kernel_weight(xc, yc, zc, raw_i, raw_j, raw_k)
                        if w <= 0.0:
                            continue
                        vel_wsum += self.vel[ii, jj, kk] * w
                        eps_wsum += self.fluid_fraction[ii, jj, kk] * w
                        w_total += w

            fluid_vel = Vector3(0.0, 0.0, 0.0)
            eps_fluid = 0.0
            if w_total > 1e-15:
                fluid_vel = vel_wsum / w_total * self.unit.dx / self.unit.dt
                eps_fluid = eps_wsum / w_total

            # 曳力 + 反作用力分配。
            if w_total > 1e-15 and 1.0 - eps_fluid > 1e-15:
                eps_p = 1.0 - eps_fluid
                d_p = 2.0 * self.dem.gf[pid].radius
                u_slip = self.dem.gf[pid].velocity - fluid_vel
                F_d = self.compute_drag_force(d_p, u_slip, eps_p)
                self.dem.gf[pid].force_fluid += F_d

                reaction_lu = -F_d * force_scale
                for raw_i in range(i_min, i_max):
                    for raw_j in range(j_min, j_max):
                        for raw_k in range(k_min, k_max):
                            ii, valid_i = self._mapping_index_x(raw_i)
                            jj, valid_j = self._mapping_index_y(raw_j)
                            kk, valid_k = self._mapping_index_z(raw_k)
                            if not (valid_i and valid_j and valid_k):
                                continue
                            if not self._is_mapping_cell(ii, jj, kk):
                                continue
                            w = self._kernel_weight(xc, yc, zc, raw_i, raw_j, raw_k)
                            if w <= 0.0:
                                continue
                            w_norm = w / w_total
                            ti.atomic_add(self.hydroforce[ii, jj, kk], reaction_lu * w_norm)

    @ti.func
    def compute_drag_force(self, diameter: float, slip: Vector3,
                           solid_fraction: float) -> Vector3:
        """返回作用在单颗粒上的 Tenneti 曳力（SI 单位）。"""
        fluid_fraction = 1.0 - solid_fraction
        drag = Vector3(0.0, 0.0, 0.0)
        if fluid_fraction > 1e-9:
            reynolds = (
                fluid_fraction * self.rho0 * diameter * tm.length(slip) / self.mu
            )
            cd0 = 1.0 + 0.15 * tm.pow(reynolds, 0.687)
            static_term = (
                5.81 * solid_fraction / (fluid_fraction ** 3)
                + 0.48 * tm.pow(solid_fraction, 1.0 / 3.0)
                / (fluid_fraction ** 4)
            )
            solid_fraction3 = solid_fraction ** 3
            dynamic_term = solid_fraction3 * reynolds * (
                0.95 + 0.61 * solid_fraction3 / (fluid_fraction ** 2)
            )
            cd_star = fluid_fraction * (
                cd0 / (fluid_fraction ** 3) + static_term + dynamic_term
            )
            drag = (
                -3.0 * tm.pi * diameter * self.mu * fluid_fraction
                * cd_star * slip
            )
        return drag

    # ------------------------------------------------------------------
    # Kernel C: 源项组装（纯局部，无邻点依赖）
    # ------------------------------------------------------------------

    @ti.kernel
    def assemble_source(self):
        """组装 Xu et al. 的质量 / 动量源，并投影到方向分布 gfield/Ffield。"""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & BOUNDARY_MASK:
                for q in ti.static(range(self.Q)):
                    self.gfield[i, j, k][q] = 0.0
                    self.Ffield[i, j, k][q] = 0.0
                continue
            Sq, Sm = self.compute_source_terms(i, j, k)
            self.compute_gF_distribution(i, j, k, Sq, Sm)

    @ti.func
    def compute_source_terms(self, i: int, j: int, k: int):
        """返回局部宏观源 ``(Sq, Sm)``。

        ``deps/dt`` 用 ``eps - prev_fluid_fraction``（格子单位 dt=1），``grad_eps``
        中心差分；``Sm = Sq·u + (hydroforce + F)/eps``。
        """
        im, valid_im = self._source_index_x(i - 1)
        ip, valid_ip = self._source_index_x(i + 1)
        jm, valid_jm = self._source_index_y(j - 1)
        jp, valid_jp = self._source_index_y(j + 1)
        km, valid_km = self._source_index_z(k - 1)
        kp, valid_kp = self._source_index_z(k + 1)

        eps = ti.max(self.fluid_fraction[i, j, k], self.MIN_FLUID_FRACTION)
        eps_im = eps
        eps_ip = eps
        eps_jm = eps
        eps_jp = eps
        eps_km = eps
        eps_kp = eps
        if valid_im and self._is_source_cell(im, j, k):
            eps_im = self.fluid_fraction[im, j, k]
        if valid_ip and self._is_source_cell(ip, j, k):
            eps_ip = self.fluid_fraction[ip, j, k]
        if valid_jm and self._is_source_cell(i, jm, k):
            eps_jm = self.fluid_fraction[i, jm, k]
        if valid_jp and self._is_source_cell(i, jp, k):
            eps_jp = self.fluid_fraction[i, jp, k]
        if valid_km and self._is_source_cell(i, j, km):
            eps_km = self.fluid_fraction[i, j, km]
        if valid_kp and self._is_source_cell(i, j, kp):
            eps_kp = self.fluid_fraction[i, j, kp]

        grad_eps = Vector3(
            0.5 * (eps_ip - eps_im),
            0.5 * (eps_jp - eps_jm),
            0.5 * (eps_kp - eps_km),
        )
        eps_rate = eps - self.prev_fluid_fraction[i, j, k]
        rho = self.rho[i, j, k]
        velocity = self.vel[i, j, k]

        Sq = -rho / eps * (eps_rate + tm.dot(velocity, grad_eps))
        Sm = Sq * velocity + (self.hydroforce[i, j, k] + self.body_force[i, j, k]) / eps
        return Sq, Sm

    @ti.func
    def compute_gF_distribution(self, i: int, j: int, k: int,
                                Sq: float, Sm: Vector3):
        """把 ``(Sq, Sm)`` 投影到离散速度方向，写入 gfield/Ffield（Eq.22/23）。"""
        rho = self.rho[i, j, k]
        velocity = self.vel[i, j, k]
        u_dot_sm = tm.dot(velocity, Sm)
        for q in ti.static(range(self.Q)):
            direction = self.c[q]
            e_dot_sm = tm.dot(direction, Sm)
            e_dot_u = tm.dot(direction, velocity)
            self.gfield[i, j, k][q] = self.w[q] * Sq
            self.Ffield[i, j, k][q] = self.w[q] * rho * (
                e_dot_sm / self.cssq
                + (e_dot_u * e_dot_sm - self.cssq * u_dot_sm)
                / (2.0 * self.cssq * self.cssq)
            )

    # ------------------------------------------------------------------
    # Kernel D: 碰撞（BGK 弛豫 + 显式方向源）
    # ------------------------------------------------------------------

    @ti.func
    def _forward_source_index(self, q: int, i: int, j: int, k: int):
        raw_i = i + ti.cast(self.c[q][0], ti.i32)
        raw_j = j + ti.cast(self.c[q][1], ti.i32)
        raw_k = k + ti.cast(self.c[q][2], ti.i32)
        ip, valid_i = self._source_index_x(raw_i)
        jp, valid_j = self._source_index_y(raw_j)
        kp, valid_k = self._source_index_z(raw_k)
        valid = valid_i and valid_j and valid_k
        if valid and (self.CT[ip, jp, kp] & BOUNDARY_MASK):
            valid = 0
        return ip, jp, kp, valid

    @ti.func
    def _omega_q(self, i: int, j: int, k: int, q: int) -> float:
        """质量源碰撞项（Eq.16/19/20，含半步修正）。"""
        ip, jp, kp, valid = self._forward_source_index(q, i, j, k)
        g_forward = 0.0
        if valid:
            g_forward = self.gfield[ip, jp, kp][q]
        return (
            self.gfield[i, j, k][q]
            + 0.5 * (g_forward - self.gfield_prev[i, j, k][q])
        )

    @ti.func
    def _omega_m(self, i: int, j: int, k: int, q: int) -> float:
        """动量源碰撞项（Eq.17/18，含半步修正）。"""
        ip, jp, kp, valid = self._forward_source_index(q, i, j, k)
        F_forward = 0.0
        if valid:
            F_forward = self.Ffield[ip, jp, kp][q]
        return (
            self.Ffield[i, j, k][q]
            + 0.5 * (F_forward - self.Ffield_prev[i, j, k][q])
        )

    @ti.func
    def _velocity_gradient(self, i: int, j: int, k: int):
        """Central-difference velocity gradient in lattice units."""
        im, _ = self._source_index_x(i - 1)
        ip, _ = self._source_index_x(i + 1)
        jm, _ = self._source_index_y(j - 1)
        jp, _ = self._source_index_y(j + 1)
        km, _ = self._source_index_z(k - 1)
        kp, _ = self._source_index_z(k + 1)

        gradient = ti.Matrix.zero(float, self.D, self.D)
        du_dx = 0.5 * (self.vel[ip, j, k] - self.vel[im, j, k])
        du_dy = 0.5 * (self.vel[i, jp, k] - self.vel[i, jm, k])
        du_dz = 0.5 * (self.vel[i, j, kp] - self.vel[i, j, km])
        for component in ti.static(range(self.D)):
            gradient[component, 0] = du_dx[component]
            gradient[component, 1] = du_dy[component]
            gradient[component, 2] = du_dz[component]
        return gradient

    @ti.func
    def _wale_mixing_length(self, i: int, j: int, k: int) -> float:
        """Paper-style WALE mixing length in lattice units."""
        wall_distance = 1.0e20
        has_wall = 0
        if ti.static(not self.periodic_x):
            if self.CT[0, j, k] & SOLID_WALL_MASK:
                wall_distance = ti.min(wall_distance, ti.cast(i, float) - 0.5)
                has_wall = 1
            if self.CT[self.Nx - 1, j, k] & SOLID_WALL_MASK:
                wall_distance = ti.min(
                    wall_distance, ti.cast(self.Nx, float) - 1.5 - ti.cast(i, float)
                )
                has_wall = 1
        if ti.static(not self.periodic_y):
            if self.CT[i, 0, k] & SOLID_WALL_MASK:
                wall_distance = ti.min(wall_distance, ti.cast(j, float) - 0.5)
                has_wall = 1
            if self.CT[i, self.Ny - 1, k] & SOLID_WALL_MASK:
                wall_distance = ti.min(
                    wall_distance, ti.cast(self.Ny, float) - 1.5 - ti.cast(j, float)
                )
                has_wall = 1
        if ti.static(not self.periodic_z):
            if self.CT[i, j, 0] & SOLID_WALL_MASK:
                wall_distance = ti.min(wall_distance, ti.cast(k, float) - 0.5)
                has_wall = 1
            if self.CT[i, j, self.Nz - 1] & SOLID_WALL_MASK:
                wall_distance = ti.min(
                    wall_distance, ti.cast(self.Nz, float) - 1.5 - ti.cast(k, float)
                )
                has_wall = 1

        mixing_length = self.WALE_CONSTANT
        if has_wall:
            mixing_length = ti.min(
                mixing_length,
                self.VON_KARMAN_CONSTANT * ti.max(wall_distance, 0.0),
            )
        return mixing_length

    @ti.func
    def _wale_viscosity(self, i: int, j: int, k: int) -> float:
        """Return the WALE sub-grid viscosity in lattice units."""
        gradient = self._velocity_gradient(i, j, k)
        strain = 0.5 * (gradient + gradient.transpose())
        gradient_sq = gradient @ gradient
        trace_gradient_sq = (
            gradient_sq[0, 0] + gradient_sq[1, 1] + gradient_sq[2, 2]
        )
        sd = 0.5 * (gradient_sq + gradient_sq.transpose())
        for axis in ti.static(range(self.D)):
            sd[axis, axis] -= trace_gradient_sq / 3.0

        strain_norm_sq = 0.0
        sd_norm_sq = 0.0
        for row in ti.static(range(self.D)):
            for column in ti.static(range(self.D)):
                strain_norm_sq += strain[row, column] ** 2
                sd_norm_sq += sd[row, column] ** 2

        denominator = (
            tm.pow(strain_norm_sq, 2.5) + tm.pow(sd_norm_sq, 1.25)
        )
        nu_turb = 0.0
        if denominator > 1.0e-30:
            mixing_length = self._wale_mixing_length(i, j, k)
            nu_turb = (
                mixing_length ** 2 * tm.pow(sd_norm_sq, 1.5) / denominator
            )
        return nu_turb

    @ti.func
    def _local_shear_relaxation(self, i: int, j: int, k: int) -> float:
        """Return local shear relaxation for none/Smagorinsky/WALE."""
        shear_relaxation = self.omega0
        is_wet_boundary = (self.CT[i, j, k] & WET_NODE_BOUNDARY_MASK) != 0
        if not is_wet_boundary:
            if ti.static(self.turbulence_model == "smagorinsky"):
                self.computeOmega(i, j, k)
                shear_relaxation = self.omega[i, j, k]
            elif ti.static(self.turbulence_model == "wale"):
                nu_effective = self.nuLu + self._wale_viscosity(i, j, k)
                tau_effective = 0.5 + nu_effective / self.cssq
                shear_relaxation = 1.0 / tau_effective
        self.omega[i, j, k] = shear_relaxation
        return shear_relaxation

    @ti.func
    def _mrt_collision(self, i: int, j: int, k: int,
                       shear_relaxation: float):
        """Return ``-M^-1 S M (f-feq)`` in velocity space."""
        f_noneq = ti.Vector.zero(float, self.Q)
        moment_noneq = ti.Vector.zero(float, self.Q)
        moment_collision = ti.Vector.zero(float, self.Q)
        omega_f = ti.Vector.zero(float, self.Q)

        for q in ti.static(range(self.Q)):
            f_noneq[q] = self.f[i, j, k][q] - self.feq[i, j, k][q]
        for moment in ti.static(range(self.Q)):
            for q in ti.static(range(self.Q)):
                moment_noneq[moment] += self.mrt_matrix[moment, q] * f_noneq[q]
            relaxation = self.mrt_relaxation[moment]
            if ti.static(5 <= moment < 10):
                relaxation = shear_relaxation
            moment_collision[moment] = -relaxation * moment_noneq[moment]
        for q in ti.static(range(self.Q)):
            for moment in ti.static(range(self.Q)):
                omega_f[q] += self.mrt_inverse[q, moment] * moment_collision[moment]
        return omega_f

    @ti.kernel
    def collide(self):
        """Selectable D3Q19 BGK/MRT relaxation plus explicit sources."""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & BOUNDARY_MASK:
                continue

            self.compute_feq(i, j, k)
            shear_relaxation = self._local_shear_relaxation(i, j, k)
            omega_f_mrt = ti.Vector.zero(float, self.Q)
            if ti.static(self.collision_model == "mrt"):
                omega_f_mrt = self._mrt_collision(
                    i, j, k, shear_relaxation
                )
            for q in ti.static(range(self.Q)):
                Omega_f = 0.0
                if ti.static(self.collision_model == "mrt"):
                    Omega_f = omega_f_mrt[q]
                else:
                    Omega_f = -shear_relaxation * (
                        self.f[i, j, k][q] - self.feq[i, j, k][q]
                    )
                Omega_q = self._omega_q(i, j, k, q)
                Omega_m = self._omega_m(i, j, k, q)
                self.fpc[i, j, k][q] = (
                    self.f[i, j, k][q] + Omega_f + Omega_q + Omega_m
                )

    # ------------------------------------------------------------------
    # 高层耦合接口
    # ------------------------------------------------------------------

    @ti.kernel
    def _initialize_history(self):
        """用初始孔隙率建立历史场，避免把初始颗粒床误视为瞬时从纯流体中生成。"""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            self.prev_fluid_fraction[i, j, k] = self.fluid_fraction[i, j, k]
            for q in ti.static(range(self.Q)):
                self.gfield_prev[i, j, k][q] = 0.0
                self.Ffield_prev[i, j, k][q] = 0.0

    @ti.kernel
    def _swap_history(self):
        """把当前孔隙率与方向源存为上一时步历史。"""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            self.prev_fluid_fraction[i, j, k] = self.fluid_fraction[i, j, k]
            self.gfield_prev[i, j, k] = self.gfield[i, j, k]
            self.Ffield_prev[i, j, k] = self.Ffield[i, j, k]

    def initialize_complete(self):
        """初始化 LBM 状态、耦合场、曳力与源项历史。"""
        self.grains2lattice()
        self._initialize_history()
        self.initialize()
        self.lattice2grains()
        self.assemble_source()
        self._swap_history()

    def update_coupling(self):
        self._swap_history()
        self.grains2lattice()
        self.lattice2grains()
        self.assemble_source()
