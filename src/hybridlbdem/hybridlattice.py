"""
Hybrid Resolved/Unresolved LBM-DEM Coupling (3D)
=================================================

架构概述
--------
本模块在同一次模拟中处理两类颗粒，每类使用最适合其尺度的耦合方法：

  粗颗粒 (dp > dx)  → 部分饱和格子法 PSC（Noble & Torczynski 1998）
                       全解析：颗粒几何形状直接映射到格子，精确处理固液界面
  细颗粒 (dp ≤ dx)  → 颗粒等效 IBM + Tenneti 曳力（Wang 2023 / Zhu 2026）
                       非解析：颗粒以固相分数分布到核函数覆盖的格子节点

耦合场布局
----------
两套命名前缀分别对应两种方法：
  coarse_*  → PSC 专用：coarse_volfrac, coarse_weight, coarse_velsolid, coarse_feqsolid
  fine_*    → IBM 专用：fine_volfrac,   fine_weight,   fine_velsolid,   fine_feqsolid

碰撞算符（按格子单元内容分派）
    纯流体         f_post = f + Ω_f
    仅粗颗粒       f_post = f + B·Ω_s² + (1-B)·Ω_f
    仅细颗粒       f_post = f + β·Ω_s¹ + (1-β)·Ω_f
    粗细共存       f_post = f + β·Ω_s¹ + B·Ω_s² + (1-β-B)·Ω_f

标准主循环调用顺序
------------------
    lattice = HybridLattice3D(Nx, Ny, Nz, omega, dx, dt, rho, dem)
    lattice.initialize_complete()

    while step < totalSteps :
        for _ in range(logSteps):
            step += 1

            lattice.prepare_step()   # 1. 颗粒→格子映射 + 细颗粒 Tenneti 权重

            lattice.collide()        # 2. 碰撞（PSC 权重 B 在内部计算，见设计说明）

            lattice.stream()         # 3. 流 + 边界条件 + rho/vel 更新（三合一）
                                     #    stream() 内部完成：
                                     #      · 各类边界传播（bounce-back、LADD、自由滑移）
                                     #      · compute_rho_vel（流体节点）
                                     #      · 湿节点 BC（Zou-He、exit、pressure、LADD inlet）
                                     #    返回时 self.vel 已是本步最新值

            lattice.lattice2grains() # 4. 力传递回 DEM（stream 后 vel 已更新，可直接使用）
                                     #    · 粗颗粒：hydroforce → 物理力/力矩
                                     #    · 细颗粒：插值 vel → Tenneti 曳力

            for _ in range(subCycles):
                demsolver.run_simulation()  # 5. DEM 子循环（使用步骤4写入的 force_fluid）

设计说明
--------
* PSC 权重 B 的延迟计算：
    B = ε(τ-½)/[(1-ε)+(τ-½)]，τ=1/ω。若启用 Smagorinsky，ω 每步在
    collide()→computeOmega() 中更新。为保证 B 与 Ω_f 使用同一步的 ω，
    B 在 collide_coarse/hybrid 内、computeOmega() 之后由
    _psc_compute_weight() 即时计算，而非在 prepare_step() 中预先计算。

* 混合区细颗粒 eps_p 的处理：
    _transfer_fine_force() 中 eps_p 含粗颗粒和细颗粒的总固相分数。
    这是有意为之：粗颗粒占据格子空间，细颗粒实际感受到更拥挤的流体环境，
    Tenneti 阻力修正应反映总固相，而非仅细颗粒固相。

参考文献
--------
  PSC 方法:   Noble & Torczynski, Int. J. Mod. Phys. C 9 (1998) 1189-1201
  Tenneti:    Tenneti et al., Int. J. Multiphase Flow 37 (2011) 1072-1092
  等效 IBM:   Wang et al., Chem. Eng. J. (2023) 142898
  镜像核函数: Zhu et al., Chem. Eng. Sci. (2026) 123562
"""

import taichi as ti
import taichi.math as tm

from src.lbm3d.lbm_solver3d import BasicLattice3D
from src.lbm3d.lbmutils import CellType
from src.dem3d.demsolver import DEMSolver

Vector3 = ti.types.vector(3, float)


# =============================================================================
# 主类
# =============================================================================

class HybridLattice3D(BasicLattice3D):
    """
    三维混合 LBM-DEM 耦合格子。

    颗粒分类规则（运行时按 radius 判断）
    ------------------------------------
      粗颗粒：2*radius > dx  → PSC 方法（coarse_* 字段）
      细颗粒：2*radius ≤ dx  → IBM/Tenneti 方法（fine_* 字段）

    字段说明
    --------
    流体属性：rho0, nuLu, nu, mu
    粗颗粒耦合：coarse_id, coarse_volfrac, coarse_weight,
               coarse_velsolid, coarse_feqsolid
    细颗粒耦合：fine_volfrac, fine_weight, fine_velsolid, fine_feqsolid,
               _fine_velsum, _fine_weightsum（内部累加器）
    力场：hydroforce（碰撞中积累的粗颗粒动量交换），hydrotorque
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

        # 流体物理/格子属性
        self.rho0 = rho
        self.omega0 = omega
        self.nuLu = (1.0 / omega - 0.5) / 3.0          # 格子动粘度
        self.nu   = self.nuLu * (dx ** 2) / dt          # 物理动粘度 [m²/s]
        self.mu   = rho * self.nu                        # 物理动力粘度 [Pa·s]

        # ------------------------------------------------------------------
        # 粗颗粒耦合字段（PSC 方法）
        # ------------------------------------------------------------------
        self.coarse_id       = ti.field(int,   shape=shape)  # 格子节点对应粒子ID，-1表示无
        self.coarse_volfrac  = ti.field(float, shape=shape)  # 固相分数 ε_c
        self.coarse_weight   = ti.field(float, shape=shape)  # PSC 权重 B（碰撞时更新）
        self.coarse_velsolid = ti.Vector.field(self.D, float, shape=shape)  # 固体速度
        self.coarse_feqsolid = ti.Vector.field(self.Q, float, shape=shape)  # 固体平衡态

        # ------------------------------------------------------------------
        # 细颗粒耦合字段（IBM/Tenneti 方法）
        # ------------------------------------------------------------------
        self.fine_volfrac    = ti.field(float, shape=shape)  # 固相分数 ε_f
        self.fine_weight     = ti.field(float, shape=shape)  # Tenneti 权重 β
        self.fine_velsolid   = ti.Vector.field(self.D, float, shape=shape)  # 固体速度
        self.fine_feqsolid   = ti.Vector.field(self.Q, float, shape=shape)  # 固体平衡态
        # 细颗粒映射内部累加器（prepare_step 期间使用）
        self._fine_velsum    = ti.Vector.field(self.D, float, shape=shape)
        self._fine_weightsum = ti.field(float, shape=shape)

        # ------------------------------------------------------------------
        # 力场（碰撞步中积累粗颗粒动量交换；lattice2grains 中读取）
        # ------------------------------------------------------------------
        self.hydroforce  = ti.Vector.field(self.D, float, shape=shape)
        self.hydrotorque = ti.Vector.field(self.D, float, shape=shape)

        # DEM 求解器引用
        self.dem = dem_solver

    # =========================================================================
    # 初始化
    # =========================================================================

    @ti.kernel
    def initialize(self):
        """将分布函数初始化为平衡态，跳过所有边界格子。"""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD
                                   | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                continue
            self.compute_feq(i, j, k)
            for q in ti.static(range(HybridLattice3D.Q)):
                self.f[i, j, k][q] = self.feq[i, j, k][q]

    # =========================================================================
    # DEM → 格子映射  ——  粗颗粒（PSC）
    # =========================================================================

    @ti.kernel
    def map_coarse_grains(self):
        """
        用 5×5×5 子格子分解将粗颗粒（dp > dx）映射到格子。

        每个格子节点记录固相分数 coarse_volfrac 和固体速度 coarse_velsolid；
        仅保留覆盖度最大的那个颗粒。

        注：PSC 权重 B 不在此处计算，而是在 collide_coarse/hybrid 内
            computeOmega() 之后由 _psc_compute_weight() 即时计算，
            以确保 B 与碰撞算符使用同一步的 ω（见模块头"设计说明"）。
        """
        self.coarse_id.fill(-1)
        self.coarse_volfrac.fill(0.0)
        self.coarse_weight.fill(0.0)
        self.coarse_velsolid.fill(0.0)

        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD
                                   | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                continue

            for gid in range(self.dem.gf.shape[0]):
                # 跳过细颗粒
                if 2.0 * self.dem.gf[gid].radius <= self.unit.dx:
                    continue

                # 颗粒中心的格子坐标
                xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin
                      + 0.5 * self.unit.dx) / self.unit.dx
                yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin
                      + 0.5 * self.unit.dx) / self.unit.dx
                zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin
                      + 0.5 * self.unit.dx) / self.unit.dx

                # 有效半径（缩短 0.5dx 用于边界格子处理）
                r    = (self.dem.gf[gid].radius - 0.5 * self.unit.dx) / self.unit.dx
                dist = ti.sqrt((xc - i)**2 + (yc - j)**2 + (zc - k)**2)
                half_diag = 0.5 * ti.sqrt(3.0)

                if dist >= r + half_diag:        # 完全在颗粒外
                    continue
                elif dist <= r - half_diag:      # 完全在颗粒内
                    self.coarse_id[i, j, k]      = gid
                    self.coarse_volfrac[i, j, k] = 1.0
                    self._psc_set_velsolid(i, j, k, gid, xc, yc, zc)
                    break                        # 一个节点只被一个颗粒主导
                else:                            # 部分覆盖：5³ 子格子积分
                    cnt = 0
                    for si in range(5):
                        for sj in range(5):
                            for sk in range(5):
                                sx = i - 0.4 + 0.2 * si
                                sy = j - 0.4 + 0.2 * sj
                                sz = k - 0.4 + 0.2 * sk
                                if ti.sqrt((xc-sx)**2 + (yc-sy)**2 + (zc-sz)**2) < r:
                                    cnt += 1
                    eps = cnt / 125.0
                    if eps > self.coarse_volfrac[i, j, k]:   # 保留覆盖度最大的颗粒
                        self.coarse_id[i, j, k]      = gid
                        self.coarse_volfrac[i, j, k] = eps
                        self._psc_set_velsolid(i, j, k, gid, xc, yc, zc)

    @ti.func
    def _psc_compute_weight(self, i: int, j: int, k: int):
        """
        计算 PSC 权重 B 并写入 coarse_weight。
        B = ε(τ-½) / [(1-ε) + (τ-½)]，使用当前步 omega（computeOmega 后调用）。
        """
        eps   = self.coarse_volfrac[i, j, k]
        tau_m = 1.0 / self.omega[i, j, k] - 0.5
        self.coarse_weight[i, j, k] = (eps * tau_m) / ((1.0 - eps) + tau_m)

    @ti.func
    def _psc_set_velsolid(self, i: int, j: int, k: int,
                          gid: int, xc: float, yc: float, zc: float):
        """计算节点 (i,j,k) 处的粗颗粒固体速度（平动 + 转动），转换为格子单位。"""
        r_vec         = Vector3(i, j, k) - Vector3(xc, yc, zc)
        omega_cross_r = tm.cross(self.dem.gf[gid].omega, r_vec * self.unit.dx)
        self.coarse_velsolid[i, j, k] = (
            (self.dem.gf[gid].velocity + omega_cross_r) * self.unit.dt / self.unit.dx
        )

    # =========================================================================
    # DEM → 格子映射  ——  细颗粒（IBM）
    # =========================================================================

    @ti.kernel
    def map_fine_grains(self):
        """
        用 Peskin 3-点 delta 核（含镜像边界修正）将细颗粒（dp ≤ dx）映射到格子。

        三遍算法（每个颗粒）：
          第1遍：denom = Σ W_bar                       （核归一化分母）
          第2遍：D_corr = Σ w_norm·(1-ε_c)              （粗颗粒占据修正分母）
          第3遍：ε_f = w_norm·(1-ε_c)/D_corr·V_grain/V_lattice

        已被粗颗粒完全占据的节点（ε_c ≥ 1.0）跳过，确保细颗粒只分布到可用空间。
        必须在 map_coarse_grains() 之后调用（依赖 coarse_volfrac）。
        """
        self.fine_volfrac.fill(0.0)
        self._fine_velsum.fill(0.0)
        self._fine_weightsum.fill(0.0)

        V_lattice = self.unit.dx ** 3
        support   = 1.5

        for gid in range(self.dem.gf.shape[0]):
            if 2.0 * self.dem.gf[gid].radius > self.unit.dx:
                continue   # 跳过粗颗粒

            xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin
                  + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin
                  + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin
                  + 0.5 * self.unit.dx) / self.unit.dx

            V_grain    = 4.0 / 3.0 * tm.pi * self.dem.gf[gid].radius ** 3
            vel_lattice = self.dem.gf[gid].velocity * self.unit.dt / self.unit.dx

            i_min = ti.max(0, ti.cast(xc - support, ti.i32))
            i_max = ti.min(self.Nx, ti.cast(xc + support + 1, ti.i32))
            j_min = ti.max(0, ti.cast(yc - support, ti.i32))
            j_max = ti.min(self.Ny, ti.cast(yc + support + 1, ti.i32))
            k_min = ti.max(0, ti.cast(zc - support, ti.i32))
            k_max = ti.min(self.Nz, ti.cast(zc + support + 1, ti.i32))

            # ------------------------------------------------------------------
            # 第1遍：核归一化分母 denom = Σ W_bar
            # ------------------------------------------------------------------
            denom = 0.0
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    for k in range(k_min, k_max):
                        if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD
                                               | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                            continue
                        if self.coarse_volfrac[i, j, k] >= 1.0:
                            continue
                        w_bar = self._kernel_with_mirror(xc, yc, zc, i, j, k)
                        if w_bar > 0.0:
                            denom += w_bar
            if denom < 1e-30:
                continue

            # ------------------------------------------------------------------
            # 第2遍：粗颗粒占据修正分母 D_corr = Σ w_norm·(1-ε_c)
            # ------------------------------------------------------------------
            denom_corr = 0.0
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    for k in range(k_min, k_max):
                        if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD
                                               | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                            continue
                        if self.coarse_volfrac[i, j, k] >= 1.0:
                            continue
                        w_bar = self._kernel_with_mirror(xc, yc, zc, i, j, k)
                        if w_bar > 0.0:
                            denom_corr += (w_bar / denom) * (1.0 - self.coarse_volfrac[i, j, k])
            if denom_corr < 1e-30:
                continue

            # ------------------------------------------------------------------
            # 第3遍：分配修正后的固相分数和速度
            #   ε_f = w_norm·(1-ε_c)/D_corr·V_grain/V_lattice
            # ------------------------------------------------------------------
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    for k in range(k_min, k_max):
                        if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD
                                               | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                            continue
                        if self.coarse_volfrac[i, j, k] >= 1.0:
                            continue
                        w_bar = self._kernel_with_mirror(xc, yc, zc, i, j, k)
                        if w_bar <= 0.0:
                            continue
                        w_norm  = w_bar / denom
                        avail   = 1.0 - self.coarse_volfrac[i, j, k]
                        eps_ij  = w_norm * avail / denom_corr * V_grain / V_lattice
                        ti.atomic_add(self.fine_volfrac[i, j, k],   eps_ij)
                        ti.atomic_add(self._fine_velsum[i, j, k],   vel_lattice * eps_ij)
                        ti.atomic_add(self._fine_weightsum[i, j, k], w_norm)

        # ------------------------------------------------------------------
        # 归一化固体速度；夹紧固相分数防止溢出
        # ------------------------------------------------------------------
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            avail = 1.0 - self.coarse_volfrac[i, j, k]
            if self.fine_volfrac[i, j, k] > avail:
                if self.fine_volfrac[i, j, k] - avail > 1e-6:
                    print("Warning: fine volfrac overflow at ({},{},{}): fine={}, coarse={}".format(
                        i, j, k, self.fine_volfrac[i, j, k], self.coarse_volfrac[i, j, k]))
                self.fine_volfrac[i, j, k] = ti.max(0.0, avail - 0.01)
            if self.fine_volfrac[i, j, k] > 1e-10:
                self.fine_velsolid[i, j, k] = (
                    self._fine_velsum[i, j, k] / self.fine_volfrac[i, j, k]
                )

    # =========================================================================
    # Tenneti 曳力权重  ——  细颗粒
    # =========================================================================

    @ti.kernel
    def compute_fine_weights(self):
        """
        用 Tenneti 模型计算每个格子节点的 IBM 权重 β。
        β = W_d = 3π·d_p^L·ν_L·(1-ε_p)·Cd(Re_p, ε_p)
        必须在 map_fine_grains() 之后调用，依赖 fine_volfrac 和 fine_velsolid。
        """
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD
                                   | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                continue
            svf = self.fine_volfrac[i, j, k]
            fvf = 1.0 - svf - self.coarse_volfrac[i, j, k]
            if svf > 0.0 and fvf > 1e-10:
                V_latt = self.unit.dx ** 3
                R_eff  = ti.pow(3.0 * V_latt * svf / (4.0 * tm.pi), 1.0 / 3.0)
                v_slip = ((self.fine_velsolid[i, j, k] - self.vel[i, j, k])
                          * self.unit.dx / self.unit.dt)
                self.fine_weight[i, j, k] = self._weight_coefficient(
                    2.0 * R_eff, v_slip, svf
                )

    @ti.func
    def _weight_coefficient(self, dp: float, u_slip: Vector3, svf: float) -> float:
        """Tenneti 模型无量纲权重 W_d = 3π·d_p^L·ν_L·(1-ε_p)·Cd。"""
        u_mag  = tm.length(u_slip)
        Re_p   = (1.0 - svf) * self.rho0 * dp * u_mag / self.mu
        Cd     = self._compute_tenneti_Cd(Re_p, svf)
        dp_L   = dp / self.unit.dx
        return 3.0 * tm.pi * dp_L * self.nuLu * (1.0 - svf) * Cd

    @ti.func
    def _compute_tenneti_Cd(self, Re_p: float, svf: float) -> float:
        """
        Tenneti 密相悬浮液曳力系数。
        Cd = (1-ε_p)·[Cd0/(1-ε_p)³ + A(ε_p) + B(Re_p, ε_p)]
          Cd0 = 1 + 0.15·Re_p^0.687
          A   = 5.81·ε_p/(1-ε_p)³ + 0.48·ε_p^(1/3)/(1-ε_p)⁴
          B   = ε_p³·Re_p·[0.95 + 0.61·ε_p³/(1-ε_p)²]
        """
        Cd  = 0.0
        fvf = 1.0 - svf
        if fvf > 1e-9:
            Cd0   = 1.0 + 0.15 * tm.pow(Re_p, 0.687)
            A_eps = (5.81 * svf / fvf**3
                     + 0.48 * tm.pow(svf, 1.0/3.0) / fvf**4)
            svf3  = svf ** 3
            B_eps = svf3 * Re_p * (0.95 + 0.61 * svf3 / fvf**2)
            Cd    = fvf * (Cd0 / fvf**3 + A_eps + B_eps)
        return Cd

    # =========================================================================
    # 碰撞步
    # =========================================================================

    @ti.kernel
    def collide(self):
        """
        混合碰撞：按格子内容分派到四种算符。

        分派逻辑（优先级）：
          粗 + 细 → collide_hybrid   （同时包含两种颗粒）
          仅粗    → collide_coarse   （PSC 动量交换）
          仅细    → collide_fine     （IBM 加权碰撞）
          无颗粒  → collide_fluid    （标准 BGK）

        PSC 权重 B 在 collide_coarse/hybrid 内部、computeOmega() 之后即时计算，
        保证 B 与 Ω_f 使用同一时刻的 ω。
        """
        self.hydroforce.fill(0.0)
        self.hydrotorque.fill(0.0)

        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD
                                   | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                continue
            self.computeOmega(i, j, k)    # 更新 ω（含 Smagorinsky 修正）
            self.compute_feq(i, j, k)     # 基于当前 vel 更新平衡态

            has_coarse = self.coarse_volfrac[i, j, k] > 0.0
            has_fine   = self.fine_volfrac[i, j, k]   > 0.0

            if has_coarse and has_fine:
                self.collide_hybrid(i, j, k)
            elif has_coarse:
                self.collide_coarse(i, j, k)
            elif has_fine:
                self.collide_fine(i, j, k)
            else:
                self.collide_fluid(i, j, k)

    # -------------------------------------------------------------------------
    # 碰撞子函数
    # -------------------------------------------------------------------------

    @ti.func
    def collide_fluid(self, i: int, j: int, k: int):
        """标准单松弛 BGK 碰撞（纯流体节点）。"""
        for q in ti.static(range(HybridLattice3D.Q)):
            self.fpc[i, j, k][q] = (
                (1.0 - self.omega[i, j, k]) * self.f[i, j, k][q]
                + self.omega[i, j, k] * self.feq[i, j, k][q]
            )

    @ti.func
    def collide_coarse(self, i: int, j: int, k: int):
        """
        PSC 碰撞算符（仅粗颗粒节点）。
        f_post = f + B·Ω_s + (1-B)·Ω_f
        动量交换积累到 hydroforce，用于 lattice2grains 中的力计算。
        """
        # 在当前步 omega 基础上计算 PSC 权重 B（延迟计算，见模块头说明）
        self._psc_compute_weight(i, j, k)
        self.compute_feq_solid_coarse(i, j, k)
        B = self.coarse_weight[i, j, k]

        for q in ti.static(range(HybridLattice3D.Q)):
            Omega_s = (self.f[i, j, k][HybridLattice3D.qinv[q]]
                       - self.feq[i, j, k][HybridLattice3D.qinv[q]]
                       + self.coarse_feqsolid[i, j, k][q]
                       - self.f[i, j, k][q])
            Omega_f = -self.omega[i, j, k] * (self.f[i, j, k][q] - self.feq[i, j, k][q])
            self.fpc[i, j, k][q] = self.f[i, j, k][q] + B * Omega_s + (1.0 - B) * Omega_f
            # 动量交换：F = -B·Ω_s·c_q（格子单位）
            self.hydroforce[i, j, k] -= B * Omega_s * HybridLattice3D.c[q]

    @ti.func
    def collide_fine(self, i: int, j: int, k: int):
        """
        IBM 加权碰撞（仅细颗粒节点）。
        f_post = f + β·Ω_s + (1-β)·Ω_f，β 来自 Tenneti 曳力模型。
        """
        self.compute_feq_solid_fine(i, j, k)
        beta = self.fine_weight[i, j, k]

        for q in ti.static(range(HybridLattice3D.Q)):
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
        混合碰撞（粗细颗粒共存节点）。
        f_post = f + β·Ω_s¹ + B·Ω_s² + (1-β-B)·Ω_f

        稳定性约束 B + β ≤ 1 在循环外裁剪，仅计算一次。
        粗颗粒动量交换积累到 hydroforce。
        """
        self._psc_compute_weight(i, j, k)
        self.compute_feq_solid_fine(i, j, k)
        self.compute_feq_solid_coarse(i, j, k)

        B    = self.coarse_weight[i, j, k]
        beta = self.fine_weight[i, j, k]

        # 稳定性裁剪：确保 B + β ≤ 1，避免分布函数发散
        if B + beta > 1.0:
            beta = 1.0 - B

        for q in ti.static(range(HybridLattice3D.Q)):
            Omega_s1 = (self.f[i, j, k][HybridLattice3D.qinv[q]]     # 细颗粒弹回算符
                        - self.feq[i, j, k][HybridLattice3D.qinv[q]]
                        + self.fine_feqsolid[i, j, k][q]
                        - self.f[i, j, k][q])
            Omega_s2 = (self.f[i, j, k][HybridLattice3D.qinv[q]]     # 粗颗粒弹回算符
                        - self.feq[i, j, k][HybridLattice3D.qinv[q]]
                        + self.coarse_feqsolid[i, j, k][q]
                        - self.f[i, j, k][q])
            Omega_f  = -self.omega[i, j, k] * (self.f[i, j, k][q] - self.feq[i, j, k][q])

            self.fpc[i, j, k][q] = (
                self.f[i, j, k][q]
                + beta * Omega_s1
                + B    * Omega_s2
                + (1.0 - B - beta) * Omega_f
            )
            # 粗颗粒动量交换（细颗粒曳力由 _transfer_fine_force 单独计算）
            self.hydroforce[i, j, k] -= B * Omega_s2 * HybridLattice3D.c[q]

    # -------------------------------------------------------------------------
    # 固体平衡态计算
    # -------------------------------------------------------------------------

    @ti.func
    def compute_feq_solid_coarse(self, i: int, j: int, k: int):
        """基于粗颗粒固体速度计算局部平衡态 feq_solid。"""
        u  = self.coarse_velsolid[i, j, k]
        uv = tm.dot(u, u)
        for q in ti.static(range(HybridLattice3D.Q)):
            cu = tm.dot(HybridLattice3D.c[q], u)
            self.coarse_feqsolid[i, j, k][q] = (
                HybridLattice3D.w[q] * self.rho[i, j, k]
                * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*uv)
            )

    @ti.func
    def compute_feq_solid_fine(self, i: int, j: int, k: int):
        """基于细颗粒固体速度计算局部平衡态 feq_solid。"""
        u  = self.fine_velsolid[i, j, k]
        uv = tm.dot(u, u)
        for q in ti.static(range(HybridLattice3D.Q)):
            cu = tm.dot(HybridLattice3D.c[q], u)
            self.fine_feqsolid[i, j, k][q] = (
                HybridLattice3D.w[q] * self.rho[i, j, k]
                * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*uv)
            )

    # =========================================================================
    # 格子 → DEM 力传递
    # =========================================================================

    @ti.kernel
    def lattice2grains(self):
        """
        统一力传递：格子 → 颗粒。

        必须在 stream() 之后调用。stream() 内部已完成：
          · 传播 + 各类边界条件（bounce-back、LADD、自由滑移、Zou-He 等）
          · compute_rho_vel（流体节点）+ 湿节点 BC
        因此 stream() 返回时 self.vel 已是本步最新值，本函数可直接插值使用。

        完整主循环：prepare_step → collide → stream → lattice2grains → dem子循环
        详见模块头"标准主循环调用顺序"。

          粗颗粒：从 hydroforce 字段（碰撞中积累的 PSC 动量交换）汇总力和力矩
          细颗粒：通过 IBM 核函数插值流体速度，用 Tenneti 模型计算曳力
        """
        self.dem.gf.force_fluid.fill(0.0)
        self.dem.gf.moment_fluid.fill(0.0)

        # 粗颗粒：PSC 动量交换 → 物理力
        for gid in range(self.dem.gf.shape[0]):
            if 2.0 * self.dem.gf[gid].radius <= self.unit.dx:
                continue
            self._transfer_coarse_force(gid)

        # 细颗粒：Tenneti 曳力
        for gid in range(self.dem.gf.shape[0]):
            if 2.0 * self.dem.gf[gid].radius > self.unit.dx:
                continue
            self._transfer_fine_force(gid)

    @ti.func
    def _transfer_coarse_force(self, gid: int):
        """
        汇总粗颗粒 gid 覆盖的所有格子节点上的 PSC 动量交换，转换为物理力。
        F_phys = F_latt · ρ · dx⁴ / dt²
        力矩：Tf = -(x_node - x_grain) × F
        """
        xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin
              + 0.5 * self.unit.dx) / self.unit.dx
        yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin
              + 0.5 * self.unit.dx) / self.unit.dx
        zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin
              + 0.5 * self.unit.dx) / self.unit.dx
        r  = self.dem.gf[gid].radius / self.unit.dx

        x0 = ti.max(0,        int(xc - r))
        x1 = ti.min(self.Nx,  int(xc + r + 2))
        y0 = ti.max(0,        int(yc - r))
        y1 = ti.min(self.Ny,  int(yc + r + 2))
        z0 = ti.max(0,        int(zc - r))
        z1 = ti.min(self.Nz,  int(zc + r + 2))

        for i in range(x0, x1):
            for j in range(y0, y1):
                for k in range(z0, z1):
                    if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD
                                           | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                        continue
                    if self.coarse_volfrac[i, j, k] > 0.0 and self.coarse_id[i, j, k] == gid:
                        scale  = self.unit.rho * self.unit.dx**4 / self.unit.dt**2
                        Ff     = self.hydroforce[i, j, k] * scale
                        r_vec  = (Vector3(xc, yc, zc) - Vector3(i, j, k)) * self.unit.dx
                        self.dem.gf[gid].force_fluid  += Ff
                        self.dem.gf[gid].moment_fluid += -tm.cross(r_vec, Ff)

    @ti.func
    def _transfer_fine_force(self, gid: int):
        """
        用 IBM 核插值计算细颗粒 gid 的 Tenneti 曳力。

        步骤：
          1. 用 _kernel_with_mirror 将流体速度插值到颗粒位置
          2. 估算颗粒位置处的有效流体分数 eps_fluid
          3. 应用 Tenneti 曳力模型

        关于 eps_p 的设计选择：
          eps_p = 1 - eps_fluid = mean(fine_volfrac + coarse_volfrac)，
          即总固相分数（含粗颗粒贡献）。
          混合区域中粗颗粒已占据格子空间，细颗粒感受到的流体环境更拥挤，
          Tenneti 阻力修正应基于总固相，而非仅细颗粒固相。
        """
        xc = (self.dem.gf[gid].position[0] - self.dem.config.domain.xmin
              + 0.5 * self.unit.dx) / self.unit.dx
        yc = (self.dem.gf[gid].position[1] - self.dem.config.domain.ymin
              + 0.5 * self.unit.dx) / self.unit.dx
        zc = (self.dem.gf[gid].position[2] - self.dem.config.domain.zmin
              + 0.5 * self.unit.dx) / self.unit.dx

        support = 1.5
        x0 = ti.max(0,        ti.cast(xc - support,     ti.i32))
        x1 = ti.min(self.Nx,  ti.cast(xc + support + 1, ti.i32))
        y0 = ti.max(0,        ti.cast(yc - support,     ti.i32))
        y1 = ti.min(self.Ny,  ti.cast(yc + support + 1, ti.i32))
        z0 = ti.max(0,        ti.cast(zc - support,     ti.i32))
        z1 = ti.min(self.Nz,  ti.cast(zc + support + 1, ti.i32))

        vel_wsum = Vector3(0.0, 0.0, 0.0)
        eps_sum  = 0.0
        w_total  = 0.0
        n_node   = 0

        for i in range(x0, x1):
            for j in range(y0, y1):
                for k in range(z0, z1):
                    if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD
                                           | CellType.FREE_SLIP | CellType.VEL_INLET_LADD):
                        continue
                    w_ij = self._kernel_with_mirror(xc, yc, zc, i, j, k)
                    if w_ij <= 0.0:
                        continue
                    vel_wsum += self.vel[i, j, k] * w_ij
                    # 流体分数：1 减去总固相（含粗颗粒），反映混合区实际流体空间
                    eps_sum  += (1.0 - self.fine_volfrac[i, j, k]
                                     - self.coarse_volfrac[i, j, k])
                    w_total  += w_ij
                    n_node   += 1

        fluid_vel = Vector3(0.0, 0.0, 0.0)
        eps_fluid = 0.0
        if w_total > 1e-15:
            fluid_vel = vel_wsum / w_total * self.unit.dx / self.unit.dt
        if n_node > 0:
            eps_fluid = eps_sum / float(n_node)

        # eps_p = 总固相分数（含粗颗粒），见上方设计说明
        eps_p = 1.0 - eps_fluid
        if eps_p > 1e-15:
            d_p   = 2.0 * self.dem.gf[gid].radius
            u_slip = self.dem.gf[gid].velocity - fluid_vel
            self.dem.gf[gid].force_fluid += self._compute_drag_force(d_p, u_slip, eps_p)

    @ti.func
    def _compute_drag_force(self, dp: float, u_slip: Vector3, svf: float) -> Vector3:
        """
        Tenneti 曳力矢量 [N]。
        F_d = -3π·d_p·μ·(1-ε_p)·Cd(Re_p, ε_p)·u_slip
        """
        u_mag  = tm.length(u_slip)
        Re_p   = (1.0 - svf) * self.rho0 * dp * u_mag / self.mu
        Cd     = self._compute_tenneti_Cd(Re_p, svf)
        return -3.0 * tm.pi * dp * self.mu * (1.0 - svf) * Cd * u_slip

    # =========================================================================
    # IBM 核函数工具
    # =========================================================================

    @ti.func
    def _threedelta(self, r: float) -> float:
        """
        Peskin 3-点径向 delta 核，支撑域 |r| ≤ 1.5 lu。

          区域1 (r < 0.5):   φ = [1 + √(1 - 3r²)] / 3
          区域2 (0.5 ≤ r ≤ 1.5): φ = [5 - 3r - √(1 - 3(1-r)²)] / 6
          区域3 (r > 1.5):   φ = 0
        """
        a = 0.0
        if r < 0.5:
            a = (1.0 + ti.sqrt(1.0 - 3.0 * r * r)) / 3.0
        elif r <= 1.5:
            a = (5.0 - 3.0 * r - ti.sqrt(1.0 - 3.0 * (1.0 - r)**2)) / 6.0
        return a

    @ti.func
    def _kernel_with_mirror(self, xc: float, yc: float, zc: float,
                            ii: int, jj: int, kk: int) -> float:
        """
        含镜像边界修正的核权重 W_bar = W(x_p) + W(x'_p)。

        实现 Zhu et al. (2026) Eq.(22)：当颗粒中心距任意壁面 < 1.5 lu 时，
        将截断的核叶片通过镜像颗粒折叠回域内，防止近壁固相分数被低估。
        六面体域的 x、y、z 三对壁面均做检查。
        """
        support = 1.5

        # 主要贡献
        dist      = ti.sqrt((xc-ii)**2 + (yc-jj)**2 + (zc-kk)**2)
        w_primary = self._threedelta(dist)
        w_mirror  = 0.0

        # x 方向壁面
        if xc < support:
            d = ti.sqrt((-xc - ii)**2 + (yc-jj)**2 + (zc-kk)**2)
            w_mirror += self._threedelta(d)
        if xc > float(self.Nx - 1) - support:
            xm = 2.0 * float(self.Nx - 1) - xc
            d  = ti.sqrt((xm-ii)**2 + (yc-jj)**2 + (zc-kk)**2)
            w_mirror += self._threedelta(d)

        # y 方向壁面
        if yc < support:
            d = ti.sqrt((xc-ii)**2 + (-yc-jj)**2 + (zc-kk)**2)
            w_mirror += self._threedelta(d)
        if yc > float(self.Ny - 1) - support:
            ym = 2.0 * float(self.Ny - 1) - yc
            d  = ti.sqrt((xc-ii)**2 + (ym-jj)**2 + (zc-kk)**2)
            w_mirror += self._threedelta(d)

        # z 方向壁面
        if zc < support:
            d = ti.sqrt((xc-ii)**2 + (yc-jj)**2 + (-zc-kk)**2)
            w_mirror += self._threedelta(d)
        if zc > float(self.Nz - 1) - support:
            zm = 2.0 * float(self.Nz - 1) - zc
            d  = ti.sqrt((xc-ii)**2 + (yc-jj)**2 + (zm-kk)**2)
            w_mirror += self._threedelta(d)

        return w_primary + w_mirror

    # =========================================================================
    # 高层接口
    # =========================================================================

    def initialize_complete(self):
        """
        仿真开始前的完整初始化序列。
        调用一次即可，之后进入主循环。
        """
        self.initialize()       # 分布函数 → 平衡态
        self.map_coarse_grains()  # 粗颗粒几何映射
        self.map_fine_grains()    # 细颗粒固相分数映射
        self.compute_fine_weights()  # 初始 Tenneti 权重

    def prepare_step(self):
        """
        每个 LBM 时间步开始时调用（在 collide 之前）。

        执行颗粒→格子映射和 Tenneti 权重更新：
          1. map_coarse_grains()    — 粗颗粒几何（PSC 权重 B 延迟到碰撞内计算）
          2. map_fine_grains()      — 细颗粒固相分数（依赖步骤1的coarse_volfrac）
          3. compute_fine_weights() — 细颗粒 Tenneti 权重 β

        完整主循环见模块头注释。
        """
        self.map_coarse_grains()
        self.map_fine_grains()
        self.compute_fine_weights()

    def update_coupling(self):
        """
        已废弃，保留用于向后兼容。

        .. deprecated::
            请改用以下两步式调用（见模块头"标准主循环调用顺序"）：
              lattice.prepare_step()    # 在 collide 之前
              ...                       # collide / stream / apply_bc / compute_macro
              lattice.lattice2grains()  # 在 compute_macro 之后
        """
        self.prepare_step()
        self.lattice2grains()

