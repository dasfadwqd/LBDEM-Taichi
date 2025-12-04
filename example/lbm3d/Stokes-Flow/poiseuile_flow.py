'''
3D Poiseuille flow in a channel.

Author: Yang Gengchao (Sun Yat-sen University)
Email: yanggch8@sysu.edu.cn
Modified for 3D simulation
'''

import os

os.system('clear')
import time
import pickle
import numpy as np

# taichi packages (set backend, default precision and device memory)
import taichi as ti

ti.init(arch=ti.gpu,
        default_fp=ti.f64,
        device_memory_fraction=0.3,
        debug=False)
SAVE_FRAMES = True
vec3 = ti.math.vec3  # 改为3D向量

# lbdem source package - 假设您已经有对应的3D类
from src.lbm3d.lbm_foecelover import ForcedLattice3D
from src.lbm3d.lbmutils import CellType


# ===================================#
# ----- User-defined Functions -----#
# ===================================#
# define the boundary conditions for the 3D channel flow
def setChannelFlow3D(lattice: ForcedLattice3D):
    CT = lattice.CT.to_numpy()
    # 设置y方向的壁面边界条件（上下壁面）
    CT[:, 0, :] = CellType.OBSTACLE  # 底部壁面
    CT[:, -1, :] = CellType.OBSTACLE  # 顶部壁面

    lattice.CT.from_numpy(CT)


# ==================================#
# ----- Parameter Declaration -----#
# ==================================#
# domain geometry and discretizations
level = 6  # 降低分辨率级别，3D计算量大
lx = 0.02  # dimension in x-direction [m]
ly = 0.01  # dimension in y-direction [m]
lz = 0.01  # dimension in z-direction [m] (新增)
Nx = int(lx / ly * (2 ** level))  # number of lattice nodes in x-direction
Ny = 2 ** level + 2  # number of lattice nodes in y-direction
Nz = 2 ** level + 2  # number of lattice nodes in z-direction (新增)
dx = ly / (2 ** level)  # lattice spacing [m]
x = np.arange(Nx) * dx - 0.5 * dx  # x-coordinates (periodic) [m]
y = np.arange(Ny) * dx - 0.5 * dx  # y-coordinates (no-slip) [m]
z = np.arange(Nz) * dx - 0.5 * dx  # z-coordinates [m] (新增)

# fluid properties
rhof = 1000.0  # fluid density [kg/m^3]
etaf = 1e-3  # dynamic viscosity [Pa s]
nu = etaf / rhof  # default kinetic viscosity [m^2/s]

# maximum flow velocity (3D channel flow)
gx = 1e-2  # gravitational acceleration [m/s^2]
# 对于3D矩形通道，最大速度公式需要修正
umax = 0.5 * rhof * gx / etaf * (0.5 * ly) ** 2  # 近似最大速度

# relaxation time and time step
tau = 0.5055  # relaxation time
omega = 1.0 / tau  # relaxation frequency
nuLU = (tau - 0.5) / 3.0  # fluid viscosity in lattice units
dt = (dx ** 2) / (nu / nuLU)  # time step [s]

# generate the 3D lattice
grav = vec3(gx, 0, 0)  # 3D gravity vector
lattice = ForcedLattice3D(Nx, Ny, Nz, omega, dx, dt, rhof, grav)  # 3D forced lattice

# flow properties
umaxLU = lattice.unit.getLbVel(umax)  # maximum velocity in lattice units
Re = umax * ly / nu  # Reynolds number
Ma = umaxLU * np.sqrt(3)  # Mach number

# error tolerance
err_tol = 1e-8  # the error tolerance
err = 1.0  # initial value for error

# iterations
step = 0  # number of cycles
totalSteps = int(500.0/dt)
logSteps = int(1/dt)

# data saving
outDir = '../Stokes-Flow/level{}/'.format(level)
os.makedirs(outDir + 'results', exist_ok=True)


# print the essential information for 3D LBM
print('*****************************************')
print('3D Domain size: {}x{}x{}'.format(Nx, Ny, Nz))
print('Lattice spacing: {} m'.format(dx))
print('Time step: {} s'.format(dt))
print('Relaxation time: {:.3f}'.format(tau))
print('Maximum velocity: {} m/s'.format(umax))
print('Reynolds number: {:.0f}'.format(Re))
print('Mach number: {}'.format(Ma))
print('*****************************************')
print('total step: {}'.format(totalSteps))
print('print step: {}'.format(logSteps))

# ====================================#
# ----- Initialization by EDFs ----- #
# ====================================#
# setup the 3D channel
setChannelFlow3D(lattice)

# initialization
lattice.initialize()

# save the initial data
results = {'t': 0,
           'vel': lattice.unit.getPhysVel(lattice.vel.to_numpy()),
           'rho': lattice.unit.getPhysRho(lattice.rho.to_numpy()),
           'p': lattice.unit.getPhysSigma((lattice.rho.to_numpy() - 1.0) / 3.0)
           }

with open(outDir + 'results/' + 'result_{:03d}.dat'.format(step // logSteps), 'wb') as fid:
    pickle.dump(results, fid)

# =============================#
# ----- LBM Calculations -----#
# =============================#
# monitor the program performance
tStart = time.perf_counter()
tLoop = time.perf_counter()
tEnd = time.perf_counter()

# store the old velocities (3D)
velOld = np.zeros((Nx, Ny - 2, Nz - 2, 3))  # 3D速度场，去除边界，3个分量



# major loop
while step < totalSteps:
    # LBM calculations
    for _ in range(logSteps):
        step += 1  # step counter
        lattice.collide()  # collision step
        lattice.stream()  # streaming step

    # store flow properties at t
    t = step * dt  # update time
    results = {'t': t,
               'vel': lattice.unit.getPhysVel(lattice.vel.to_numpy()),
               'rho': lattice.unit.getPhysRho(lattice.rho.to_numpy()),
               'p': lattice.unit.getPhysSigma((lattice.rho.to_numpy() - 1.0) / 3.0)
               }

    with open(outDir + 'results/' + 'result_{:03d}.dat'.format(step // logSteps), 'wb') as fid:
        pickle.dump(results, fid)

    # stop running if the error is smaller than the threshold
    vel = lattice.vel.to_numpy()[:, 1:-1, 1:-1, :]
    err = np.sum(np.sqrt((vel - velOld) ** 2)) / (Nx * (Ny - 2) * (Nz - 2)) / umaxLU
    if err < err_tol:
        break

    # update the old velocities
    velOld = vel.copy()

    # print performance and error
    tEnd = time.perf_counter()  # pause time counting
    dtLoop = tEnd - tLoop  # time difference between two logs
    dtTotal = tEnd - tStart  # time spent since the start
    mlups = Nx * Ny * Nz * logSteps / dtLoop / 1e6  # 3D million lattice updates per second
    print(
        "Step: {} | Error: {:.2e} | Speed: {:.0f} MLU/s | Total time: {:.0f} seconds".format(step, err, mlups, dtTotal))
    tLoop = time.perf_counter()



# =========================#
# ----- Finalization -----#
# =========================#
# print the overall performance
tEnd = time.perf_counter()
dtTotal = tEnd - tStart
mlups = Nx * Ny * Nz * step / dtTotal / 1e6  # 3D性能计算
print("Final Step: {} | Error: {:.2e} | Speed: {:.0f} MLU/s | Total time: {:.0f} seconds".format(step, err, mlups,
                                                                                                 dtTotal))

