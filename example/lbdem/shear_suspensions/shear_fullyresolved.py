'''
Rheology of sheared suspensions .
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
# define the boundary conditions for flow past a fixed cylinder
def setLidDrivenCavity(lattice: PSCLattice3D, uwLU):
    for i in range(lattice.Nx):
        for j in range(lattice.Ny):
            for k in range(lattice.Nz):
                if k == 0:  # bottom wall
                    lattice.CT[i, j, k] = CellType.VEL_LADD
                    lattice.vel[i, j, k][0] = -uwLU
                if k == lattice.Nz - 1:  # top wall
                    lattice.CT[i, j, k] = CellType.VEL_LADD
                    lattice.vel[i, j, k][0] = uwLU




# ==================================#
# ----- Parameter Declaration -----#
# ==================================#

# domain geometry and discretizations
lx = 0.01  # dimension in x-direction [m]
ly = 0.004  # dimension in y-direction [m]
lz = 0.01  # dimension in z-direction [m]
dia = 0.0005
dx = dia / 8  # lattice spacing [m]
Nx = int( lx / dx )+ 2  # number of lattice nodes in x-direction
Ny = int( ly / dx )+ 2  # number of lattice nodes in y-direction
Nz = int( lz / dx )+ 2  # number of lattice nodes in y-direction
x = np.arange(Nx) * dx - 0.5 * dx  # x-coordinates [m]
y = np.arange(Ny) * dx - 0.5 * dx  # y-coordinates [m]
z = np.arange(Nz) * dx - 0.5 * dx  # z-coordinates [m]



dens = 2650  # particle density [kg/m3]

# fluid properties
rho = 1000 # fluid density [kg/m^3]
# flow velocity at the entrance and flow regime
umax = 1
cp = 5
# DEM simulation parameters
particle_init = 'shear_5.p4p'


grav = Vector3(0.0, 0.0 , 0.0 )                          # reduced gravity due to buoyancy [m/s^2]

r = 100
nu = 1e-6  # fluid kinematic viscosity [m^2/s]
Re = rho * (dia/2)**2 * r / nu
# LBM relaxation time and time step
tau = 0.85  # relaxation time
omega = 1.0 / tau  # relaxation frequency
nuLU = (tau - 0.5) / 3.0  # fluid viscosity in lattice units
dtLBM = (dx ** 2) / (nu / nuLU)  # time step [s]


# iterations
step = 0  # number of cycles
total_time = 1
totalSteps = round(total_time / dtLBM)  # total number of time step
logSteps = round(0.001 / dtLBM)  # print log info every 'logSteps' steps
subCycles = 10  # number of sub-cycles (no influence if no collision!)
dtDEM = dtLBM / subCycles  # DEM time step [s]
# error tolerance
err_tol = 1e-9  # the error tolerance
err = 1.0  # initial value for error
# data saving
if SAVE_RESULTS:
    outDir = '../shear_suspensions/fully_cp{}/'.format(cp)
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

contact_model = HertzContactConfig(
                pp_friction=0.0,
                pw_friction=0.0,
                pp_restitution=1.0,
                pw_restitution=1.0

                )
# Set up particle properties
config = DEMSolverConfig(
        domain=domain,
        dt=dtDEM,
        gravity=grav,
        contact_model=contact_model
    )

config.set_particle_properties(
    elastic_modulus=7e10,
    poisson_ratio=0.3
)
config.set_wall_properties(
    elastic_modulus=7e10,
    poisson_ratio=0.3
)

# Initialize solver
domain_min = Vector3(xmin , ymin ,zmin)
domain_max = Vector3(xmax , ymax ,zmax)
demsolver = DEMSolver(config)
demsolver.init_particle_fields(particle_init, domain_min, domain_max)
demsolver.set_contact_model("hertz")

print(config.summary())

# Print spatial partitioning info for debugging
print(f"Hash table size = {demsolver.bpcd.hash_table.shape[0]}, cell_size = {demsolver.bpcd.cell_size}")
# ===========================================#
# ----- Initialization LBM Simulation ----- #
# ===========================================#
# generate the lattice
lattice = PSCLattice3D(Nx, Ny, Nz, omega, dx, dtLBM, rho ,demsolver)  # basic lattice
umaxLU = lattice.unit.getLbVel(umax*0.5)  # terminal velocity in lattice units
# set boundary conditions
setLidDrivenCavity(lattice, umaxLU)
# initialization
lattice.initialize()

plane_xy1, plane_xz1, plane_yz1 = lattice.compute_plane_average_stress(axis=2, position=Nz - 2)
plane_xy2, plane_xz2, plane_yz2 = lattice.compute_plane_average_stress(axis=2, position=1)
tao_xy = lattice.unit.getPhysSigma(plane_xy1 + plane_xy2)
nu_new = 0.5 * tao_xy / r

print(f"xy平面剪切应力: τ_xy={tao_xy:.6f}")
print(f"表观粘度： nu1/nu={nu_new / nu:.6f}")
# save the initial data
if SAVE_RESULTS:
    results = {'t': 0,
               'velf': lattice.unit.getPhysVel(lattice.vel.to_numpy()),
               'rhof': lattice.unit.getPhysRho(lattice.rho.to_numpy()),
               'pf': lattice.unit.getPhysSigma((lattice.rho.to_numpy() - 1.0) / 3.0),
               'τ_xy' :tao_xy,
               'nu1/nu' : nu_new

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
# store the old velocities
vel_old = np.zeros((Nx, Ny, Nz, 3))
# major loop
while  step < totalSteps:
    # LBM calculations
    for _ in range(logSteps):
        # LBM calculations
        step += 1  # step counter
        lattice.collide()  # collision step
        lattice.stream()  # streaming step
        lattice.compute_stress()

        # DEM calculations
        for _ in range(subCycles):
            demsolver.run_simulation()  # inter-particle interactions

    # store flow properties at t
    if SAVE_RESULTS:
        results = {'t': step * dtLBM,
                   'velf': lattice.unit.getPhysVel(lattice.vel.to_numpy()),
                   'rhof': lattice.unit.getPhysRho(lattice.rho.to_numpy()),
                   'pf': lattice.unit.getPhysSigma((lattice.rho.to_numpy() - 1.0) / 3.0),
                   'τ_xy': tao_xy,
                   'nu1/nu': nu_new

                   }

        with open(outDir + 'results/' + 'result_{:03d}.dat'.format(step // logSteps), 'wb') as fid:
            pickle.dump(results, fid)
        demsolver.save_single(p4p_file, p4c_file, step * dtLBM)
        plane_xy1, plane_xz1, plane_yz1 = lattice.compute_plane_average_stress(axis=2, position=Nz - 2)
        plane_xy2, plane_xz2, plane_yz2 = lattice.compute_plane_average_stress(axis=2, position=1)
        tao_xy = lattice.unit.getPhysSigma(plane_xy1 + plane_xy2)
        nu_new = 0.5 * tao_xy / r

        print(f"xy平面剪切应力: τ_xy={tao_xy:.6f}")
        print(f"表观粘度： nu1/nu={nu_new / nu:.6f}")


        # calculate the error
        vel_arr = lattice.unit.getPhysVel(lattice.vel.to_numpy())
        err = np.sqrt(np.mean(vel_arr - vel_old) ** 2) / umax
        if err < err_tol: break

        # update the old velocity
        vel_old = vel_arr.copy()

    # print performance and error
    tEnd = time.perf_counter()  # pause time counting
    dtLoop = tEnd - tLoop  # time difference between two logs
    dtTotal = tEnd - tStart  # time spent since the start
    mlups = Nx * Ny * Nz * logSteps / dtLoop / 1e6  # million lattice updates per second
    print("Step: {}/{} | Error: {:.3e} | Speed: {:.0f} MLU/s | Total time: {:.0f} seconds".format(step, totalSteps,
                                                                                                  err,
                                                                                                  mlups, dtTotal))

    tLoop = time.perf_counter()

    # ==========================#
    # ----- Finalization ----- #
    # ==========================#
    # print the overall performance
    tEnd = time.perf_counter()
    dtTotal = tEnd - tStart
    mlups = Nx * Ny * Nz * step / dtTotal / 1e6
    print("Step: {}/{} | Error: {:.3e} | Speed: {:.0f} MLU/s | Total time: {:.0f} seconds".format(step, totalSteps, err,
                                                                                                  mlups, dtTotal))