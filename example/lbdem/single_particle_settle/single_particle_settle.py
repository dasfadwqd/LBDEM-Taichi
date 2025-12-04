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
from src.dem3d.demslover import DEMSolver, DEMSolverConfig

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
lx = 0.1  # dimension in x-direction [m]
ly = 0.16  # dimension in y-direction [m]
lz = 0.1  # dimension in z-direction [m]
dia = 0.015
dx = dia / 16  # lattice spacing [m]
Nx = int( lx / dx )+2  # number of lattice nodes in x-direction
Ny = int( ly / dx )+2  # number of lattice nodes in y-direction
Nz = int( lz / dx )+2  # number of lattice nodes in y-direction
x = np.arange(Nx) * dx - 0.5 * dx  # x-coordinates [m]
y = np.arange(Ny) * dx - 0.5 * dx  # y-coordinates [m]
z = np.arange(Nz) * dx - 0.5 * dx  # z-coordinates [m]

# fluid properties
rho = 960.0  # fluid density [kg/m^3]
mu = 58e-3  # fluid dynamic viscosity [Pa s]
nu = mu/rho  # fluid kinematic viscosity [m^2/s]

dens = 1120  # particle density [kg/m3]

# flow velocity at the entrance and flow regime
Re = 31.9 # Reynolds number
umax = 12.8e-2


# DEM simulation parameters
particle_init = 'single_particle.p4p'
stiffness = 1e5                                                # contact stiffness [N/m]
dp_ratio = 0.2                                                  # critical damping ratio
fric = 0.0                                                      # friction coefficient
contact_model = 'linear'
grav = Vector3(0.0, -9.81*(dens-rho)/dens , 0.0)                          # reduced gravity due to buoyancy [m/s^2]
# LBM relaxation time and time step
tau = 0.60  # relaxation time
omega = 1.0 / tau  # relaxation frequency
nuLU = (tau - 0.5) / 3.0  # fluid viscosity in lattice units
dtLBM = (dx ** 2) / (nu / nuLU)  # time step [s]


# iterations
step = 0  # number of cycles
total_time = 3.0
totalSteps = round(total_time / dtLBM)  # total number of time step
logSteps = round(0.05 / dtLBM)  # print log info every 'logSteps' steps
subCycles = 10  # number of sub-cycles (no influence if no collision!)
dtDEM = dtLBM / subCycles  # DEM time step [s]

# data saving
if SAVE_RESULTS:
    outDir = '../single_particle_settle/Re{}_fully/'.format(Re)
    os.makedirs(outDir + 'results', exist_ok=True)

# =======================================#
# ----- Initialize DEM Simulation ----- #
# =======================================#
# instantiate DEM simulation
config = DEMSolverConfig(xmin=np.min(x) + 0.5 * dx,
                 xmax=np.max(x) - 0.5 * dx,
                 ymin=np.min(y) + 0.5 * dx,
                 ymax=np.max(y) - 0.5 * dx,
                 zmin=np.min(z) + 0.5 * dx,
                 zmax=np.max(z) - 0.5 * dx,
                 dt=dtDEM,
                 grav=grav)

# Set up particle properties
config.set_particle_properties(
    init_particles=particle_init,
    elastic_modulus=1e8,
    poisson_ratio=0.2,
    stiffness_normal=stiffness,
    stiffness_tangential=stiffness,
    damping_normal=dp_ratio,
    damping_tangential=dp_ratio
)


# Set up contact properties for collision handling
config.set_contact_properties(
    pp_friction=fric,  # Particle-particle friction
    pw_friction=fric,  # Particle-wall friction
    pp_restitution=0.9,  # Particle-particle restitution
    pw_restitution=0.9,  # Particle-wall restitution
    contact_model=contact_model,  # Contact model type
    set_max_coordinate_number=10  # Max contacts per particle
)

# Initialize solver
domain_min = config.set_domain_min
domain_max = config.set_domain_max
demsolver = DEMSolver(config)
demsolver.init_particle_fields(config.set_init_particles, domain_min, domain_max)
demsolver.set_contact_model(config.contact_model)

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
               'Fy': demsolver.gf[0].force_fluid[1],
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
print('DEM info')
print('Domain extend: x({}, {}), y({}, {}) , z({}, {}) '.format(demsolver.config.xmin, demsolver.config.xmax, demsolver.config.ymin,
                                                                demsolver.config.ymax, demsolver.config.zmin, demsolver.config.zmax))
print('Time step: {} s'.format(demsolver.config.dt))
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
                   'Fy': demsolver.gf[0].force_fluid[1],
                    }

        with open(outDir + 'results/' + 'result_{:03d}.dat'.format(step // logSteps), 'wb') as fid:
            pickle.dump(results, fid)
        demsolver.save_single(p4p_file, p4c_file, step * dtLBM)

    # print performance and error
    tEnd = time.perf_counter()  # pause time counting
    dtLoop = tEnd - tLoop  # time difference between two logs
    dtTotal = tEnd - tStart  # time spent since the start
    mlups = Nx * Ny *  Nz * logSteps / dtLoop / 1e6  # million lattice updates per second
    print("Step: {}/{} | Velocity ratio: {:.4f} | Speed: {:.0f} MLU/s | Total time: {:.0f} seconds".format(step,
                                                                                                           totalSteps,
                                                                                                           -demsolver.gf[0].velocity[1] / umax,
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
print("Step: {}/{} | Velocity ratio: {:.3f} | Speed: {:.0f} MLU/s | Total time: {:.0f} seconds".format(step,
                                                                                                           totalSteps,
                                                                                                           -demsolver.gf[0].velocity[1] / umax,
                                                                                                           mlups,
                                                                                                           dtTotal))
