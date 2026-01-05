import taichi as ti
import sys
import os
import time


# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Taichi packages (set backend, default precision and device memory)
import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64)

# Source packages
from src.dem3d.demsolver import DEMSolver
from src.dem3d.demconfig import DEMSolverConfig , DomainBounds, LinearContactConfig ,HertzContactConfig

# Initialize Taichi context
ti.init(arch=ti.gpu, debug=False)

# =====================================
# Type Definitions
# =====================================

Vector3 = ti.types.vector(3,float)
Vector3i = ti.types.vector(3, int)

# =====================================
# Simulation Constants
# =====================================
# Domain boundaries
xmin, xmax = 0, 0.01  # Domain boundaries in x-direction
ymin, ymax = 0, 0.005  # Domain boundaries in y-direction
zmin, zmax = 0, 0.005  # Domain boundaries in z-direction

# Physical properties
grav = Vector3(1.0, 0.0, -5)  # Gravitational acceleration (m/s²)
dt = 1e-7  # Time step size (seconds)
target_time = 0.5  # Total simulation time (seconds)
saving_interval_time = 0.01  # Time interval between saved frames (seconds)

domain = DomainBounds(xmin , xmax,
                      ymin , ymax,
                      zmin , zmax)

contact_model = HertzContactConfig(
                pp_friction=0.0,
                pw_friction=0.0,
                pp_restitution=1.0,
                pw_restitution=1.0

                )



# Particle material properties
particle_init = "grainfreefall.p4p"  # Initial particle information file



def main():
    """Main function to run the Discrete Element Method (DEM) simulation."""
    # Initialize solver configuration with domain boundaries and physical parameters
    config = DEMSolverConfig(
        domain=domain,
        dt=dt,
        gravity=grav,
        contact_model=contact_model
    )

    config.set_particle_properties(
        elastic_modulus=7e10,
        poisson_ratio=0.3,
        max_coordinate_number = 64
    )
    config.set_wall_properties(
        elastic_modulus=7e10,
        poisson_ratio=0.3
    )



    # Initialize solver
    domain_min = Vector3(xmin , ymin ,zmin)
    domain_max = Vector3(xmax , ymax ,zmax)
    solver = DEMSolver(config)
    solver.init_particle_fields(particle_init, domain_min, domain_max)
    solver.set_contact_model("hertz")

    print(config.summary())

    # Print spatial partitioning info for debugging
    print(f"Hash table size = {solver.bpcd.hash_table.shape[0]}, cell_size = {solver.bpcd.cell_size}")

    # Initialize simulation
    step = 0
    elapsed_time = 0.0
    start_time = time.time()

    nsteps = int(target_time / dt)
    saving_interval_steps = int(saving_interval_time / dt)

    # Open output files for particle positions and contacts
    with open('output.p4p', encoding="UTF-8", mode='w') as p4p, \
            open('output.p4c', encoding="UTF-8", mode='w') as p4c:
        solver.save_single(p4p, p4c, solver.config.dt * step)
        # Main simulation loop
        while step < nsteps:
            for _ in range(saving_interval_steps):
                step += 1
                elapsed_time += dt
                solver.run_simulation()


            # Print progress information
            progress_percentage = step / nsteps * 100
            print(f"Solved steps: {step} / {nsteps} ({progress_percentage:.2f}%)")

            # Save current state
            solver.save_single(p4p, p4c, solver.config.dt * step)

    # Calculate and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")


if __name__ == '__main__':
    main()