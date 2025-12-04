'''
Solve the problem of lid driven cavity flow.

'''

import os

os.system('clear')
import time
import pickle
import numpy as np

# taichi packages (set backend, default precision and device memory)
import taichi as ti

ti.init(arch=ti.gpu,
        debug=False)
SAVE_FRAMES = True

# source package
from src.lbm3d.lbm_solver3d import BasicLattice3D
from src.lbm3d.lbmutils import CellType




# ===================================#
# ----- User-defined Functions -----#
# ===================================#
# define the boundary conditions for flow past a fixed cylinder
def setLidDrivenCavity(lattice: BasicLattice3D, uwLU):
    for i in range(lattice.Nx):
        for j in range(lattice.Ny):
            for k in range(lattice.Nz):
                if i == 0:  # left wall
                    lattice.CT[i, j ,k] = CellType.OBSTACLE
                elif i == lattice.Nx - 1:  # right wall
                    lattice.CT[i, j ,k] = CellType.OBSTACLE
                elif j == 0:  # bottom wall
                    lattice.CT[i, j ,k] = CellType.OBSTACLE
                elif j == lattice.Ny - 1:  # top wall
                    lattice.CT[i, j ,k] = CellType.VEL_LADD
                    lattice.vel[i, j,k][0] = uwLU
                elif k == 0:     # back wall
                    lattice.CT[i, j ,k] = CellType.OBSTACLE
                elif k == lattice.Nz - 1: # front wall
                    lattice.CT[i, j, k] = CellType.OBSTACLE


# ==================================#
# ----- Parameter Declaration -----#
# ==================================#
# domain geometry and discretizations
lx = 0.1  # dimension in x-direction [m]
ly = 0.1  # dimension in y-direction [m]
lz = 0.1  # dimension in z-direction [m]
dx = lx / 64  # lattice spacing [m]
Nx = 64 + 2  # number of lattice nodes in x-direction
Ny = 64 + 2  # number of lattice nodes in y-direction
Nz = 64 + 2  # number of lattice nodes in y-direction
x = np.arange(Nx) * dx - 0.5 * dx  # x-coordinates [m]
y = np.arange(Ny) * dx - 0.5 * dx  # y-coordinates [m]
z = np.arange(Nz) * dx - 0.5 * dx  # y-coordinates [m]

# fluid properties
rho = 1000.0  # fluid density [kg/m^3]
mu = 0.001  # fluid dynamic viscosity [Pa s]
nu = mu/rho  # fluid kinematic viscosity [m^2/s]

# flow velocity at the entrance and flow regime
Re = 400  # Reynolds number
uw = Re*nu/lx  # maximum velocity at lid [m/s]

# relaxation time and time step
tau = 0.5023  # relaxation time
omega = 1.0 / tau  # relaxation frequency
nuLU = (tau - 0.5) / 3.0  # fluid viscosity in lattice units
dt = (dx ** 2) / (nu / nuLU)  # time step [s]

# generate the lattice
lattice = BasicLattice3D(Nx, Ny, Nz, omega, dx, dt, rho)  # basic lattice
uwLU = lattice.unit.getLbVel(uw)  # entrance velocity in lattice units

# error tolerance
err_tol = 1e-9  # the error tolerance
err = 1.0  # initial value for error

# iterations
step = 0  # number of cycles
totalSteps = round(5000.0 / dt)  # total number of time step
logSteps = round(100 / dt)  # print log info every 'logSteps' steps

# data saving
outDir = '../lid_driven_cavity_flow/Re{}/'.format(Re)

if SAVE_FRAMES:
    os.makedirs(outDir + 'results', exist_ok=True)


# print the essential information for LBM
print('*****************************************')
print('Domain size: {}x{}x{}'.format(Nx, Ny, Nz))
print('Lattice spacing: {} m'.format(dx))
print('Time step: {} s'.format(dt))
print('Lid velocity: {} m/s'.format(uw))
print('Reynolds number: {:.0f}'.format(Re))
print('Mach number: {}'.format(uwLU * np.sqrt(3)))
print('Relaxation time: {:.3f}'.format(tau))
print('Total time step: {}'.format(totalSteps))
print('Output info every {} steps'.format(logSteps))
print('*****************************************')

# ========================================#
# ----- Initialization by the EDFs ----- #
# ========================================#
# set boundary conditions
setLidDrivenCavity(lattice, uwLU)
# initialization
lattice.initialize()
# save the initial data
results = {'t': 0,
           'vel': lattice.unit.getPhysVel(lattice.vel.to_numpy()),
           'rho': lattice.unit.getPhysRho(lattice.rho.to_numpy()),
           'p': lattice.unit.getPhysSigma((lattice.rho.to_numpy() - 1.0) / 3.0)}

with open(outDir + 'results/' + 'result_{:03d}.dat'.format(step // logSteps), 'wb') as fid:
    pickle.dump(results, fid)

# =============================#
# ----- LBM Calculations -----#
# =============================#
# monitor the program performance
tStart = time.perf_counter()
tLoop = time.perf_counter()
tEnd = time.perf_counter()

# store the old velocities
vel_old = np.zeros((Nx, Ny, Nz, 3))


# major loop
while  step < totalSteps:
    # LBM calculations
    for _ in range(logSteps):
        step += 1  # step counter
        lattice.collide()  # collision step
        lattice.stream()  # streaming step
        #visualizer.display_gui('velocity' )

    # store flow properties at t
    t = step * dt  # update time
    results = {'t': t,
               'vel': lattice.unit.getPhysVel(lattice.vel.to_numpy()),
               'rho': lattice.unit.getPhysRho(lattice.rho.to_numpy()),
               'p': lattice.unit.getPhysSigma((lattice.rho.to_numpy() - 1.0) / 3.0)}

    #visualizer.save_image("lid_driven_cavity_flow",'density')

    with open(outDir + 'results/' + 'result_{:03d}.dat'.format(step // logSteps), 'wb') as fid:
        pickle.dump(results, fid)

    # calculate the error
    vel_arr = lattice.unit.getPhysVel(lattice.vel.to_numpy())
    err = np.sqrt(np.mean(vel_arr - vel_old) ** 2) / uw
    if err < err_tol: break

    # update the old velocity
    vel_old = vel_arr.copy()

    # print performance and error
    tEnd = time.perf_counter()  # pause time counting
    dtLoop = tEnd - tLoop  # time difference between two logs
    dtTotal = tEnd - tStart  # time spent since the start
    mlups = Nx * Ny * Nz * logSteps / dtLoop / 1e6  # million lattice updates per second
    print("Step: {}/{} | Error: {:.3e} | Speed: {:.0f} MLU/s | Total time: {:.0f} seconds".format(step, totalSteps, err,
                                                                                                  mlups, dtTotal))
    tLoop = time.perf_counter()

# =========================#
# ----- Finalization -----#
# =========================#
# print the overall performance
tEnd = time.perf_counter()
dtTotal = tEnd - tStart
mlups = Nx * Ny * Nz * step / dtTotal / 1e6
print(
    "Step: {}/{} | Error:{:.3e} | Speed: {:.0f} MLU/s | Total time: {:.0f} seconds".format(step, totalSteps, err, mlups,
                                                                                           dtTotal))