'''
simulations on a single sphere settling under gravity
Reference: https://doi.org/10.1063/1.1512918
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
        default_ip=ti.i32,
        debug=False)
SAVE_RESULTS = True

# source package
from src.lbdem.psclattice3d import PSCLattice3D
from src.lbm3d.lbmutils import CellType
from src.dem3d.demsolver import DEMSolver
from src.dem3d.demconfig import DEMSolverConfig , DomainBounds, LinearContactConfig ,HertzContactConfig

Vector3 = ti.types.vector(3, float)


# ===================================#
# ----- User-defined Functions -----#
# ===================================#
def setContainer(lattice: PSCLattice3D):
    for i in range(lattice.Nx):
        for j in range(lattice.Ny):
            for k in range(lattice.Nz):
                if (i == 0) or (i == lattice.Nx - 1) or (j == 0)\
                        or (j == lattice.Ny - 1) or (k == 0) or (k == Nz -1):
                    lattice.CT[i, j ,k] = CellType.OBSTACLE


# ==================================#
# ----- Parameter Declaration -----#
# ==================================#
# domain geometry and discretizations
lx = 0.01  # dimension in x-direction [m]
ly = 0.01  # dimension in y-direction [m]
lz = 0.04  # dimension in z-direction [m]
dia = 1/600
dx = dia / 16  # lattice spacing [m]
Nx = int( lx / dx )+2  # number of lattice nodes in x-direction
Ny = int( ly / dx )+2  # number of lattice nodes in y-direction
Nz = int( lz / dx )+2  # number of lattice nodes in y-direction
x = np.arange(Nx) * dx - 0.5 * dx  # x-coordinates [m]
y = np.arange(Ny) * dx - 0.5 * dx  # y-coordinates [m]
z = np.arange(Nz) * dx - 0.5 * dx  # z-coordinates [m]



dens = 1140  # particle density [kg/m3]



# fluid properties
rho =1000 # fluid density [kg/m^3]
mu = 1e-3  # fluid dynamic viscosity [Pa s]
nu = mu/rho  # fluid kinematic viscosity [m^2/s]
# flow velocity at the entrance and flow regime

umax =0.08
Re = 70
# DEM simulation parameters
particle_init = 'dkt.p4p'


grav = Vector3(0.0, 0.0 , -9.81*(dens-rho)/dens)                          # reduced gravity due to buoyancy [m/s^2]
# LBM relaxation time and time step
tau = 0.62  # relaxation time
omega = 1.0 / tau  # relaxation frequency
nuLU = (tau - 0.5) / 3.0  # fluid viscosity in lattice units
dtLBM = (dx ** 2) / (nu / nuLU)  # time step [s]


# iterations
step = 0  # number of cycles
total_time = 0.7
totalSteps = round(total_time / dtLBM)  # total number of time step
logSteps = round(0.025 / dtLBM)  # print log info every 'logSteps' steps
subCycles = 50  # number of sub-cycles (no influence if no collision!)
dtDEM = dtLBM / subCycles  # DEM time step [s]

# data saving
if SAVE_RESULTS:
    outDir = '../dkt/'
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


contact_model = LinearContactConfig(
    stiffness_normal=5000,
    stiffness_tangential=1428,
    damping_normal=0.2,
    damping_tangential=0.2,
    pp_friction=0.2,
    pw_friction=0.2
 )
# Set up particle properties
config = DEMSolverConfig(
        domain=domain,
        dt=dtDEM,
        gravity=grav,
        contact_model=contact_model
    )

config.set_particle_properties(
    elastic_modulus=1e8,
    poisson_ratio=0.3,
    max_coordinate_number = 64
)
config.set_wall_properties(
    elastic_modulus=1e8,
    poisson_ratio=0.3
)

# Initialize solver
domain_min = Vector3(xmin , ymin ,zmin)
domain_max = Vector3(xmax , ymax ,zmax)
demsolver = DEMSolver(config)
demsolver.init_particle_fields(particle_init, domain_min, domain_max)
demsolver.set_contact_model("linear")

print(config.summary())

# Print spatial partitioning info for debugging
print(f"Hash table size = {demsolver.bpcd.hash_table.shape[0]}, cell_size = {demsolver.bpcd.cell_size}")
# ===========================================#
# ----- Initialization LBM Simulation ----- #
# ===========================================#
# generate the lattice
lattice = PSCLattice3D(Nx, Ny, Nz, omega, dx, dtLBM, rho ,demsolver)  # basic lattice
umaxLU = lattice.unit.getLbVel(umax)  # terminal velocity in lattice units
# set boundary conditions
setContainer(lattice)
# initialization
lattice.initialize()

# save the initial data
if SAVE_RESULTS:
    results = {'t': 0,
               'velf': lattice.unit.getPhysVel(lattice.vel.to_numpy()),
               'rhof': lattice.unit.getPhysRho(lattice.rho.to_numpy()),
               'pf': lattice.unit.getPhysSigma((lattice.rho.to_numpy() - 1.0) / 3.0),

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
        # LBM calculations
        step += 1  # step counter
        lattice.collide()  # collision step
        lattice.stream()  # streaming step

        # DEM calculations
        for _ in range(subCycles):
            demsolver.run_simulation()  # inter-particle interactions


    # store flow properties at t
    if SAVE_RESULTS:
        results = {'t': step * dtLBM,
                    'velf': lattice.unit.getPhysVel(lattice.vel.to_numpy()),
                    'rhof': lattice.unit.getPhysRho(lattice.rho.to_numpy()),
                    'pf': lattice.unit.getPhysSigma((lattice.rho.to_numpy() - 1.0) / 3.0),

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
