'''
Numerical model for the seepage-induced fine-particle migration through the coarse skeleton formed of three coarse particles
Reference: https://doi.org/10.1016/j.compgeo.2019.02.002
'''

import os
os.system('clear')
import time
import pickle
import numpy as np

# taichi packages (set backend, default precision and device memory)
import taichi as ti
import math

ti.init(arch=ti.gpu,
        default_fp=ti.f64,
        default_ip=ti.i32,


        debug=False)
SAVE_RESULTS = True

# source package
from src.hybridlbdem.hybridlattice import HybridLattice3D
from src.lbm3d.lbmutils import CellType
from src.dem3d.demsolver import DEMSolver
from src.dem3d.demconfig import DEMSolverConfig , DomainBounds, LinearContactConfig ,HertzContactConfig

Vector3 = ti.types.vector(3, float)


# ===================================#
# ----- User-defined Functions -----#
# ===================================#
def setContainer(lattice: HybridLattice3D ,uwLU):
    for i in range(lattice.Nx):
        for j in range(lattice.Ny):
            for k in range(lattice.Nz):
                if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
                    lattice.CT[i, j, k] = CellType.OBSTACLE
                elif k == 0:


                    lattice.CT[i, j, k] = CellType.VEL_INLET_LADD | CellType.BACK
                    lattice.vel[i, j, k] = [0.0, 0.0, uwLU]

                elif k == Nz - 1:
                    # 同样，角点已被上面的墙壁占据
                    # lattice.CT[i, j, k] = CellType.Pre_ZOUHE | CellType.FRONT
                    # lattice.rho[i, j, k] = 1.0

                    lattice.CT[i, j, k] = CellType.VEL_EXIT | CellType.FRONT



# ==================================#
# ----- Parameter Declaration -----#
# ==================================#
# domain geometry and discretizations
lx = 0.04  # dimension in x-direction [m]
ly = 0.04  # dimension in y-direction [m]
lz = 0.220  # dimension in z-direction [m]

dx = 0.0015  # lattice spacing [m]
Nx = int( lx / dx )+2  # number of lattice nodes in x-direction
Ny = int( ly / dx )+2  # number of lattice nodes in y-direction
Nz = int( lz / dx )+2  # number of lattice nodes in y-direction
x = np.arange(Nx) * dx - 0.5 * dx  # x-coordinates [m]
y = np.arange(Ny) * dx - 0.5 * dx  # y-coordinates [m]
z = np.arange(Nz) * dx - 0.5 * dx  # z-coordinates [m]



dens = 2450  # particle density [kg/m3]


# fluid properties
rho = 1000 # fluid density [kg/m^3]
mu = 1e-3  # fluid dynamic viscosity [Pa s]
nu = mu/rho  # fluid kinematic viscosity [m^2/s]
# flow velocity at the entrance and flow regime
Re = 1.5 # Reynolds number
umax = 0.011

# DEM simulation parameters
particle_init = 'hybridcf.p4p'


grav = Vector3(0.0, 0.0 , -9.81*(dens-rho)/dens)                          # reduced gravity due to buoyancy [m/s^2]
# LBM relaxation time and time step
tau = 0.50026  # relaxation time
omega = 1.0 / tau  # relaxation frequency
nuLU = (tau - 0.5) / 3.0  # fluid viscosity in lattice units
dtLBM = (dx ** 2) / (nu / nuLU)  # time step [s]


# iterations
step = 0  # number of cycles
total_time = 20
totalSteps = round(total_time / dtLBM)  # total number of time step
logSteps = round(0.001 / dtLBM)  # print log info every 'logSteps' steps
subCycles = 200  # number of sub-cycles (no influence if no collision!)
dtDEM = dtLBM / subCycles  # DEM time step [s]

# data saving
if SAVE_RESULTS:
    outDir = '../hybridexperiment/'
    os.makedirs(outDir + 'results', exist_ok=True)

# =======================================#
# ----- Initialize DEM Simulation ----- #
# =======================================#
# instantiate DEM simulation

xmin=np.min(x) + 0.5 * dx,
xmax=np.max(x) - 0.5 * dx,
ymin=np.min(y) + 0.5 * dx,
ymax=np.max(y) - 0.5 * dx,
zmin=np.min(z) + 0.5 * dx,
zmax=np.max(z) - 0.5 * dx,
domain = DomainBounds(xmin=np.min(x) + 0.5 * dx,
        xmax=np.max(x) - 0.5 * dx,
        ymin=np.min(y) + 0.5 * dx,
        ymax=np.max(y) - 0.5 * dx,
        zmin=np.min(z) + 0.5 * dx,
        zmax=np.max(z) - 0.5 * dx,
                      )

# Set up particle properties
contact_model = HertzContactConfig(
                pp_restitution=0.9,
                pw_restitution=0.9,
                pp_friction=0.1545,
                pw_friction=0.1333
                )
# Set up particle properties
config = DEMSolverConfig(
        domain=domain,
        dt=dtDEM,
        gravity=grav,
        contact_model=contact_model
    )

config.set_particle_properties(
        elastic_modulus=5e10,
        poisson_ratio=0.25,
        max_coordinate_number = 128
    )
config.set_wall_properties(
    elastic_modulus=2e10,
    poisson_ratio=0.3
)

#config.set_periodic_boundaries(x_periodic=True,z_periodic=True)
# Initialize solver
domain_min = Vector3(xmin , ymin ,zmin)
domain_max = Vector3(xmax , ymax ,zmax)
demsolver = DEMSolver(config)
demsolver.init_particle_fields(particle_init, domain_min, domain_max)
demsolver.set_contact_model("hertz")
for i in range(37464):
    demsolver.gf[i].freeze = True
print(config.summary())

# Print spatial partitioning info for debugging
print(f"Hash table size = {demsolver.bpcd.hash_table.shape[0]}, cell_size = {demsolver.bpcd.cell_size}")
# ===========================================#
# ----- Initialization LBM Simulation ----- #
# ===========================================#
# generate the lattice
lattice = HybridLattice3D(Nx, Ny, Nz, omega, dx, dtLBM, rho ,demsolver)  # basic lattice
umaxLU = lattice.unit.getLbVel(umax)  # terminal velocity in lattice units

# initialization
lattice.initialize_complete()
# set boundary conditions
setContainer(lattice, umaxLU)

# save the initial data
if SAVE_RESULTS:
    results = {'t': 0,
               'velf': lattice.unit.getPhysVel(lattice.vel.to_numpy()),
               'rhof': lattice.unit.getPhysRho(lattice.rho.to_numpy()),
               'pf': lattice.unit.getPhysSigma((lattice.rho.to_numpy() - 1.0) / 3.0),
               'omega': lattice.omega.to_numpy()

               }

    with open(outDir + 'results/' + 'result_{:03d}.dat'.format(step // logSteps), 'wb') as fid:
        pickle.dump(results, fid)

    p4p_file = open(outDir + 'output.p4p', encoding="UTF-8", mode='w')
    p4c_file = open(outDir + 'output.p4c', encoding="UTF-8", mode='w')


# print the essential information for LBM
print('*****************************************')
print('LBM info')
print('Domain size: {}x{}x{}'.format(Nx, Ny ,Nz))
print('Lattice spacing: {} m'.format(dx))
print('LBM time step: {} s'.format(dtLBM))
print('Relaxation time: {:.3f}'.format(tau))
print('Reynolds number: {:.3f}'.format(Re))
print('Mach number: {}'.format(umaxLU * np.sqrt(3)))
print('-----------------------')
print('-----------------------')
print('Simulation info')
print('Total steps: {}'.format(totalSteps))
print('Save data every {} steps'.format(logSteps))
print('*****************************************')

# ==============================#
# ----- LBM Calculations ----- #
# ==============================#
# monitor the program performance
tStart = time.perf_counter()
tLoop = time.perf_counter()
tEnd = time.perf_counter()

# major loop
while  step < totalSteps:
    # LBM calculations
    for _ in range(logSteps):
        step += 1

        lattice.prepare_step()  # 1. 颗粒→格子映射 + 细颗粒 Tenneti 权重

        lattice.collide()  # 2. 碰撞（PSC 权重 B 在内部计算，见设计说明）

        lattice.stream()  # 3. 流 + 边界条件 + rho/vel 更新（三合一）


        lattice.lattice2grains()

        for _ in range(subCycles):
            demsolver.run_simulation()



    # store flow properties at t
    if SAVE_RESULTS:
        results = {'t': step * dtLBM,
                    'velf': lattice.unit.getPhysVel(lattice.vel.to_numpy()),
                    'rhof': lattice.unit.getPhysRho(lattice.rho.to_numpy()),
                    'pf': lattice.unit.getPhysSigma((lattice.rho.to_numpy() - 1.0) / 3.0),
                   'omega': lattice.omega.to_numpy()

                    }

        with open(outDir + 'results/' + 'result_{:03d}.dat'.format(step // logSteps), 'wb') as fid:
            pickle.dump(results, fid)
        demsolver.save_single(p4p_file, p4c_file, step * dtLBM)

    # print performance and error
    tEnd = time.perf_counter()  # pause time counting
    dtLoop = tEnd - tLoop  # time difference between two logs
    dtTotal = tEnd - tStart  # time spent since the start
    mlups = Nx * Ny *  Nz * logSteps / dtLoop / 1e6  # million lattice updates per second
    print("Step: {}/{} | Speed: {:.0f} MLU/s | Total time: {:.0f} seconds".format(step,
                                                                                  totalSteps,
                                                                                  mlups,
                                                                                  dtTotal))

    tLoop = time.perf_counter()

# ==========================#
# ----- Finalization ----- #
# ==========================#
# print the overall performance
tEnd = time.perf_counter()
dtTotal = tEnd - tStart
mlups = Nx * Ny * Nz * step / dtTotal / 1e6
print("Step: {}/{} | Speed: {:.0f} MLU/s | Total time: {:.0f} seconds".format(step,
                                                                                         totalSteps,

                                                                                         mlups,
                                                                                         dtTotal))
