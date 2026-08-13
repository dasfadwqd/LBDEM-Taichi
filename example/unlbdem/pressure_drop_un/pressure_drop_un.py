"""
packed_bed_sim.py
多孔介质压力降模拟（非解析 resolved LBM-DEM）

文献依据（Vlogman et al., PoF 37, 033305, 2025, Section IV A）

颗粒文件命名:
    vf0.06.p4p  vf0.10.p4p  vf0.19.p4p  vf0.28.p4p  vf0.35.p4p
    （与 packed_bed_sim.py 放在同一目录）
"""

import os
import sys
import time
import pickle
import math
import json
import numpy as np

# ============================================================
# 日志：Tee 同时输出到屏幕和文件
# ============================================================
class _Tee:
    def __init__(self, stream, filepath):
        self._screen = stream
        self._file = open(filepath, 'w', buffering=1, encoding='utf-8')

    def write(self, data):
        self._screen.write(data)
        self._file.write(data)

    def flush(self):
        self._screen.flush()
        self._file.flush()

    def fileno(self):
        return self._screen.fileno()

    def close(self):
        self._file.close()

    def __getattr__(self, name):
        return getattr(self._screen, name)

_tee_handle = None
_t0_wall = time.perf_counter()

def setup_tee(log_path: str):
    global _tee_handle
    _tee_handle = _Tee(sys.stdout, log_path)
    sys.stdout = _tee_handle

def log(msg: str):
    elapsed = time.perf_counter() - _t0_wall
    print(f'[{elapsed:8.1f}s] {msg}', flush=True)

# ============================================================
# 初始化 Taichi
# ============================================================
os.system('clear')
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f64, default_ip=ti.i32, debug=False)

from src.unlbdem.eqlattice import EqIMBlattice3D
from src.lbm3d.lbmutils import CellType
from src.dem3d.demsolver import DEMSolver
from src.dem3d.demconfig import DEMSolverConfig, DomainBounds, LinearContactConfig

Vector3 = ti.types.vector(3, float)

# ============================================================
# 体积分数
# ============================================================
VALID_VF = [0.06, 0.10, 0.19, 0.28, 0.35]

if len(sys.argv) >= 2:
    target_vf = float(sys.argv[1])
else:
    target_vf = 0.35

target_vf = round(target_vf, 2)
if target_vf not in VALID_VF:
    raise ValueError(f"体积分数 {target_vf} 不在支持列表 {VALID_VF} 中")

# dx / Dp 比例（argv[2]，默认 1.0）。示例: python pressure_drop_un.py 0.28 0.9
dx_ratio = 1.0

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
particle_init = os.path.join(BASE_DIR, f'vf{target_vf:.2f}.p4p')
RESULT_DIR = os.path.join(BASE_DIR, f'results_vf{target_vf:.2f}')

if not os.path.exists(particle_init):
    raise FileNotFoundError(f"找不到颗粒文件: {particle_init}")

os.makedirs(os.path.join(RESULT_DIR, 'results'), exist_ok=True)

# 挂载 Tee
_run_ts = time.strftime('%Y%m%d_%H%M%S')
_log_path = os.path.join(RESULT_DIR, f'run_{_run_ts}.log')
setup_tee(_log_path)
print(f'[日志] 输出同步保存至 {_log_path}', flush=True)

# ============================================================
# 边界条件设置
# ============================================================
def setContainer(lattice: EqIMBlattice3D, uwLU):
    for i in range(lattice.Nx):
        for j in range(lattice.Ny):
            for k in range(lattice.Nz):
                if i == 0:
                    lattice.CT[i, j, k] = CellType.VEL_INLET_LADD | CellType.LEFT
                    lattice.vel[i, j, k][0] = uwLU
                elif i == lattice.Nx - 1:
                    #lattice.CT[i, j, k] = CellType.VEL_EXIT | CellType.RIGHT
                    lattice.CT[i, j, k] = CellType.Pre_ZOUHE | CellType.RIGHT
                    lattice.rho[i, j, k] = 1.0

# ============================================================
# 几何参数（文献 Section IV A）
# ============================================================
Dp = 0.002          # 颗粒直径 [m]
Lx = 30 * Dp        # 0.060 m
Ly = 10 * Dp        # 0.020 m
Lz = 10 * Dp        # 0.020 m

# 压降采样截面（x=0.1L 和 x=0.9L，间距名义 0.8L）
# 注意: 实际站点会吸附到最近网格节点，压降梯度的分母用吸附后站距 actual_DL，
#       不能直接用名义 DL（dx 不能整除 0.8L 时会引入偏差）。
x_lo = 0.1 * Lx     # 0.006 m
x_hi = 0.9 * Lx     # 0.054 m
DL = x_hi - x_lo    # 0.048 m（= 0.8L，仅作名义参考）

# ============================================================
# 网格
# ============================================================
dx = Dp * dx_ratio
Nx = int(Lx / dx) + 2
Ny = int(Ly / dx) + 2
Nz = int(Lz / dx) + 2

x = np.arange(Nx) * dx - 0.5 * dx
y = np.arange(Ny) * dx - 0.5 * dx
z = np.arange(Nz) * dx - 0.5 * dx

# 网格元数据：供后处理画图读取，避免画图脚本硬编码几何/网格
# 实际物理流体域: 剔除 x/z 两侧各 1 个 ghost 后的长度 = (N-2)*dx
grid_meta = {
    'dx': dx,
    'nx': Nx, 'ny': Ny, 'nz': Nz,
    'dp': Dp,
    'vf': target_vf,
    'domain_m': [0.0, (Nx - 2) * dx, 0.0, (Ny - 2) * dx, 0.0, (Nz - 2) * dx],
}
with open(os.path.join(RESULT_DIR, 'grid_meta.json'), 'w', encoding='utf-8') as _fj:
    json.dump(grid_meta, _fj, indent=2)

# ============================================================
# 流体参数
# ============================================================
rho_f = 1000          # 流体密度 [kg/m³]
nu = 18e-6            # 运动粘度 [m²/s]
Re = 0.8              # 文献规定
U_in = Re * nu / Dp   # 入口速度 [m/s]

# 固定时间步长（文献 Vlogman et al. 2025 参考值），松弛参数由此导出
dtLBM = 2.e-4 * Lx / U_in
nuLU = nu * dtLBM / dx ** 2
tau = 3.0 * nuLU + 0.5
omega = 1.0 / tau

U_LU = U_in * dtLBM / dx
Ma = U_LU * math.sqrt(3)
assert Ma < 0.1, f"Ma={Ma:.4f} 过大，请调整参数"
assert 0.5 < tau < 2.0, f"tau={tau:.4f} 超出稳定范围 (0.5, 2.0)，请调整 dx_ratio"

# ============================================================
# 时间控制
# ============================================================
total_time = 500.0
totalSteps = round(total_time / dtLBM)
logSteps = max(1, round(0.5 / dtLBM))
subCycles = 1
dtDEM = dtLBM / subCycles

N_WINDOW = 10
STEADY_TOL = 0.05

# ============================================================
# 启动摘要
# ============================================================
print('=' * 62)
print(f'  多孔介质压降模拟（非解析）| ε_p = {target_vf:.2f}')
print(f'  网格       : {Nx}×{Ny}×{Nz}，dx = {dx*1e3:.3f} mm（= Dp × {dx_ratio}）')
print(f'  Re         : {Re}，U_in = {U_in:.4f} m/s，ν = {nu:.1e} m²/s')
print(f'  dtLBM      : {dtLBM:.4e} s（固定）→ tau = {tau:.4f}，omega = {omega:.4f}，nuLU = {nuLU:.4f}')
print(f'  Ma         : {Ma:.5f}')
print(f'  logSteps   : {logSteps}（每 {logSteps*dtLBM:.3f} s 记录）')
print(f'  稳态判据   : 近 {N_WINDOW} 点振荡 < {STEADY_TOL*100:.0f}% 均值')
print(f'  结果目录   : {RESULT_DIR}')
print('=' * 62, flush=True)

# ============================================================
# 初始化 DEM
# ============================================================
log('初始化 DEM ...')
grav = Vector3(0.0, 0.0, 0.0)

domain = DomainBounds(
    xmin=np.min(x) + 0.5 * dx,
    xmax=np.max(x) - 0.5 * dx,
    ymin=np.min(y) + 0.5 * dx,
    ymax=np.max(y) - 0.5 * dx,
    zmin=np.min(z) + 0.5 * dx,
    zmax=np.max(z) - 0.5 * dx,
)

contact_model = LinearContactConfig(
    stiffness_normal=4.5e2,
    stiffness_tangential=2.5e2,
    damping_normal=0.3,
    damping_tangential=0.2,
    pp_friction=0.15,
    pw_friction=0.15,
)

config = DEMSolverConfig(
    domain=domain,
    dt=dtDEM,
    gravity=grav,
    contact_model=contact_model,
)
config.set_particle_properties(elastic_modulus=1e8, poisson_ratio=0.3, max_coordinate_number=64)
config.set_wall_properties(elastic_modulus=1e8, poisson_ratio=0.3)
config.set_periodic_boundaries(x_periodic=False, y_periodic=True, z_periodic=True)

demsolver = DEMSolver(config)
demsolver.init_particle_fields(
    particle_init,
    Vector3(domain.xmin, domain.ymin, domain.zmin),
    Vector3(domain.xmax, domain.ymax, domain.zmax),
)
demsolver.set_contact_model("linear")

n_particles = demsolver.gf.shape[0]
for i in range(n_particles):
    demsolver.gf[i].freeze = True

log(f'DEM 初始化完成，颗粒数: {n_particles}')
print(config.summary(), flush=True)

# ============================================================
# 初始化 LBM
# ============================================================
log('初始化 LBM 格子 ...')
lattice = EqIMBlattice3D(Nx, Ny, Nz, omega, dx, dtLBM, rho_f, demsolver)
U_LU_bc = lattice.unit.getLbVel(U_in)

lattice.initialize_complete()
setContainer(lattice, U_LU_bc)
log('LBM 格子初始化完成')

# 验证 BC
print(f'  [验证] 入口 CT : {lattice.CT[0, Ny//2, Nz//2]} '
      f'（期望 = {CellType.VEL_ZOUHE | CellType.LEFT}）')
print(f'  [验证] 入口速度: {lattice.vel[0, Ny//2, Nz//2][0]:.6e} LU '
      f'（期望 ≈ {U_LU_bc:.6e}）', flush=True)

# ============================================================
# 压降计算
# ============================================================
# 采样站点吸附到最近网格节点。节点间实际距离 (i_hi-i_lo)*dx 才是压力
# 实际降落的长度；名义 0.8L 仅在 dx 恰好整除时才精确，不能作为分母。
i_lo = int(np.argmin(np.abs(x - x_lo)))
i_hi = int(np.argmin(np.abs(x - x_hi)))
x_lo_snap = x[i_lo]
x_hi_snap = x[i_hi]
actual_DL = x_hi_snap - x_lo_snap   # = (i_hi - i_lo) * dx

# 物理声速
cs_phys = dx / (math.sqrt(3) * dtLBM)

print(f'  [压降换算] cs_phys={cs_phys:.3f} m/s')
print(f'  [压降换算] i_lo={i_lo} x={x_lo_snap*1e3:.3f}mm (目标 {x_lo*1e3:.1f}mm), '
      f'i_hi={i_hi} x={x_hi_snap*1e3:.3f}mm (目标 {x_hi*1e3:.1f}mm)')
print(f'  [压降换算] 实际站距 = {actual_DL*1e3:.3f} mm（名义 0.8L = {DL*1e3:.1f} mm，'
      f'偏差 {(actual_DL-DL)/DL*100:+.2f}%）', flush=True)


def calc_pressure_drop(rho_np: np.ndarray, debug: bool = False):
    """
    按截面平均压力差计算压降梯度。
    无量纲化采用粘性标度：Π = (ΔP/ΔL) * Dp² / (η0 * U_in)
    """
    rho_lo = np.mean(rho_np[i_lo, :, :])
    rho_hi = np.mean(rho_np[i_hi, :, :])

    drho_lu = rho_lo - rho_hi
    dp_lu = drho_lu / 3.0

    dp_phys = lattice.unit.getPhysSigma(dp_lu)
    dpdl_phys = dp_phys / actual_DL

    # 粘性标度无量纲化
    eta0 = rho_f * nu  # 动力粘度 [Pa·s]
    dpdl_nd = dpdl_phys * Dp ** 2 / (eta0 * U_in)

    if debug:
        print(f'    [DEBUG] rho_lo={rho_lo:.8f}, rho_hi={rho_hi:.8f}')
        print(f'            drho_lu={drho_lu:.3e}, dp_lu={dp_lu:.3e}')
        print(f'            dp_phys={dp_phys:.6e} Pa, dpdl={dpdl_phys:.6e} Pa/m')
        print(f'            dpdl_nd={dpdl_nd:.6e} (粘性标度)', flush=True)
    return dpdl_phys, dpdl_nd


def is_steady(history: list) -> bool:
    if len(history) < N_WINDOW:
        return False
    w = np.array(history[-N_WINDOW:])
    mean_w = np.mean(w)
    if mean_w == 0:
        return False
    return (np.max(w) - np.min(w)) / abs(mean_w) < STEADY_TOL


# ============================================================
# 初始状态保存
# ============================================================
log('保存初始状态 ...')
step = 0
log_idx = 0
hist_dp = []

rho0 = lattice.rho.to_numpy()
dp0_p, dp0_nd = calc_pressure_drop(rho0)

outDir = RESULT_DIR + '/'

results = {
    't': 0,
    'meta': grid_meta,
    'velf': lattice.unit.getPhysVel(lattice.vel.to_numpy()),
    'rhof': lattice.unit.getPhysRho(rho0),
    'pf': lattice.unit.getPhysSigma((rho0 - 1.0) / 3.0),
    'volfrac': lattice.volfrac.to_numpy(),   # <-- 新增
    'weight': lattice.weight.to_numpy(),
}
with open(outDir + 'results/result_000.dat', 'wb') as fid:
    pickle.dump(results, fid)

csv_path = os.path.join(RESULT_DIR, 'pressure_drop_history.csv')
csv_f = open(csv_path, 'w')
csv_f.write('step,t_phys[s],dpdl_phys[Pa/m],dpdl_nd[-]\n')
csv_f.write(f'0,0.000000e+00,{dp0_p:.6e},{dp0_nd:.6e}\n')
csv_f.flush()

# 流体力输出目录
ff_dir = os.path.join(RESULT_DIR, 'fluid_force')
os.makedirs(ff_dir, exist_ok=True)

# 初始流体力（全零）
with open(os.path.join(ff_dir, 'ff_000000.txt'), 'w') as f_ff:
    f_ff.write('TIMESTEP  PARTICLES\n')
    f_ff.write(f'{0.0:.6f} {n_particles}\n')
    f_ff.write('ID  FFX  FFY  FFZ  FFMAG\n')
    for i_p in range(n_particles):
        f_ff.write(f'{i_p+1}  {0.0:.6e}  {0.0:.6e}  {0.0:.6e}  {0.0:.6e}\n')

log('进入主循环 ...')
print('=' * 62, flush=True)

# ============================================================
# 主循环
# ============================================================
tStart = time.perf_counter()
tLoop = time.perf_counter()
steady = False

while step < totalSteps and not steady:
    for _ in range(logSteps):
        step += 1
        for _ in range(subCycles):
            demsolver.run_simulation()
        lattice.collide()
        lattice.stream()
        lattice.update_coupling()
        rho_np = lattice.rho.to_numpy()
        nan_count = np.sum(~np.isfinite(rho_np))
        if nan_count > 0:
            print(f'[WARN] step={step}: {nan_count} 个格子出现 NaN/Inf')

    log_idx += 1
    t_phys = step * dtLBM

    rho_np = lattice.rho.to_numpy()
    vel_np = lattice.vel.to_numpy()

    debug_flag = (log_idx <= 5)
    if debug_flag:
        print(f'    [DEBUG] rho全场 min={rho_np[1:-1,1:-1,1:-1].min():.8f}, '
              f'max={rho_np[1:-1,1:-1,1:-1].max():.8f}, '
              f'mean={rho_np[1:-1,1:-1,1:-1].mean():.8f}')
        vel_mag = np.sqrt((vel_np[1:-1,1:-1,1:-1,:]**2).sum(axis=-1))
        vx_mean = vel_np[1:-1,1:-1,1:-1,0].mean()
        print(f'    [DEBUG] vel_x均值={vx_mean:.6e} LU, '
              f'速度场最大={vel_mag.max():.6e} LU', flush=True)

    dpdl_p, dpdl_nd = calc_pressure_drop(rho_np, debug=debug_flag)
    hist_dp.append(dpdl_p)
    steady = is_steady(hist_dp)

    # 保存流场数据
    results = {
        't': t_phys,
        'meta': grid_meta,
        'velf': lattice.unit.getPhysVel(vel_np),
        'rhof': lattice.unit.getPhysRho(rho_np),
        'pf': lattice.unit.getPhysSigma((rho_np - 1.0) / 3.0),
        'volfrac': lattice.volfrac.to_numpy(),  # <-- 新增
        'weight': lattice.weight.to_numpy(),
    }
    with open(outDir + f'results/result_{log_idx:03d}.dat', 'wb') as fid:
        pickle.dump(results, fid)

    # 保存流体力
    with open(os.path.join(ff_dir, f'ff_{step:06d}.txt'), 'w') as f_ff:
        f_ff.write('TIMESTEP  PARTICLES\n')
        f_ff.write(f'{t_phys:.6f} {n_particles}\n')
        f_ff.write('ID  FFX  FFY  FFZ  FFMAG\n')
        for i_p in range(n_particles):
            f_f = demsolver.gf[i_p].force_fluid
            ff_mag = math.sqrt(f_f[0]**2 + f_f[1]**2 + f_f[2]**2)
            f_ff.write(f'{i_p+1}  {f_f[0]:.6e}  {f_f[1]:.6e}  {f_f[2]:.6e}  {ff_mag:.6e}\n')

    csv_f.write(f'{step},{t_phys:.6e},{dpdl_p:.6e},{dpdl_nd:.6e}\n')
    csv_f.flush()

    tNow = time.perf_counter()
    mlups = Nx * Ny * Nz * logSteps / (tNow - tLoop) / 1e6
    tLoop = tNow
    tag = '  ★ STEADY' if steady else ''
    print(f'[ε={target_vf:.2f}] '
          f'step={step:7d}/{totalSteps} | '
          f't={t_phys:8.2f}s | '
          f'ΔP/L={dpdl_p:10.4f} Pa/m  nd={dpdl_nd:.4f} | '
          f'{mlups:6.0f} MLU/s | '
          f'elapsed={tNow-tStart:6.0f}s{tag}',
          flush=True)

# ============================================================
# 收尾
# ============================================================
csv_f.close()

w_arr = np.array(hist_dp[-N_WINDOW:] if len(hist_dp) >= N_WINDOW else hist_dp)
ss_dpdl = float(np.mean(w_arr))

# 稳态无量纲化（粘性标度）
eta0 = rho_f * nu
ss_dpdl_nd = ss_dpdl * Dp ** 2 / (eta0 * U_in)

with open(os.path.join(RESULT_DIR, 'steady_state.txt'), 'w') as fh:
    fh.write(f'vf,{target_vf}\n')
    fh.write(f'n_particles,{n_particles}\n')
    fh.write(f'steps_run,{step}\n')
    fh.write(f't_phys_s,{step * dtLBM:.4f}\n')
    fh.write(f'converged,{"yes" if steady else "no"}\n')
    fh.write(f'dpdl_phys_Pa_per_m,{ss_dpdl:.6e}\n')
    fh.write(f'dpdl_nd_viscous_scale,{ss_dpdl_nd:.6e}\n')

tTotal = time.perf_counter() - tStart
reason = '收敛（稳态）' if steady else '达到时间上限'
print('=' * 62)
log(f'完成 | ε_p={target_vf:.2f} | {reason}')
print(f'  稳态 ΔP/L = {ss_dpdl:.4f} Pa/m  （粘性标度无量纲 = {ss_dpdl_nd:.4f}）')
print(f'  总步数 {step}，物理时间 {step * dtLBM:.2f} s，耗时 {tTotal:.1f} s')
print('=' * 62, flush=True)

if _tee_handle is not None:
    sys.stdout = _tee_handle._screen
    _tee_handle.close()
    print(f'[日志] 完整日志已保存至: {_log_path}')