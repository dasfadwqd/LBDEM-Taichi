import taichi as ti
import taichi.math as tm
import os
import numpy as np
import time
import math

from .utils import *
from ..bpcd import BPCD
from ..demconfig import DEMSolverConfig, HertzContactConfig, LinearContactConfig
from ..dateclass import Grain, Material, Wall, Contact
from ..contactmanager import ContactModel, HertzMindlinContactModel, LinearContactModel

Vector3 = ti.types.vector(3, float)
Vector4 = ti.types.vector(4, float)


#=====================================
# Environmental Variables
#=====================================
DoublePrecisionTolerance: float = 1e-12 # Boundary between zeros and non-zeros


@ti.data_oriented
class DEMSolver:
    def __init__(self, config: DEMSolverConfig):
        self.config = config
        # Broad phase collision detection
        self.bpcd: BPCD
        # Material mapping
        self.mf: ti.StructField  # Material types: 0 - particles; 1 - walls
        # Particle fields
        self.gf: ti.StructField
        self.cf: ti.StructField  # neighbors for every particle
        self.cfn: ti.StructField  # neighbor counter for every particle

        self.wf: ti.StructField
        self.wcf: ti.StructField

        self.cp: ti.StructField  # collision pairs
        self.cn: ti.SNode  # collision pair node

        self.contact_model = None  # 接触模型

    def save(self, time: float):
        '''
        save the solved data at <time> to file .p4p and .p4c
        '''
        out_dir = "output"
        os.makedirs(out_dir, exist_ok=True)
        idx = int(round(time / self.config.dt))

        file_name = os.path.join(out_dir, f"output_T{idx:06d}")

        with open(file_name + ".p4p", "w", encoding="UTF-8") as p4p, \
                open(file_name + ".p4c", "w", encoding="UTF-8") as p4c:
            self.save_single(p4p, p4c, time)

    def save_single(self, p4pfile, p4cfile, t: float):
        '''
        save the solved data at <time> to <p4pfile> and <p4cfile>
        usage:
            p4p = open('output.p4p',encoding="UTF-8",mode='w')
            p4c = open('output.p4c',encoding="UTF-8",mode='w')
            while(True):
                solver.save_single(p4p, p4c, elapsed_time)
        '''
        tk1 = time.time()

        # P4P file for particles
        n = self.gf.shape[0]


        p4pfile.write("TIMESTEP  PARTICLES\n")
        p4pfile.write(f"{t} {n}\n")
        p4pfile.write("ID  GROUP  RAD  MASS  PX  PY  PZ  VX  VY  VZ\n")


        np_ID = self.gf.ID.to_numpy()
        np_groupID = self.gf.groupID.to_numpy()
        np_radius = self.gf.radius.to_numpy()
        np_mass = self.gf.mass.to_numpy()
        np_position = self.gf.position.to_numpy()
        np_velocity = self.gf.velocity.to_numpy()


        data = np.column_stack([
            np_ID, np_groupID, np_radius, np_mass,
            np_position[:, 0], np_position[:, 1], np_position[:, 2],
            np_velocity[:, 0], np_velocity[:, 1], np_velocity[:, 2]
        ])
        np.savetxt(p4pfile, data, fmt='%d %d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e')

        # P4C file for contacts
        np_i = self.cf.i.to_numpy()
        np_j = self.cf.j.to_numpy()
        np_position = self.cf.position.to_numpy()
        np_force_a = self.cf.force_a.to_numpy()
        np_active = self.cf.isActive.to_numpy()


        active_mask = np_active.astype(bool)
        n_active = np.sum(active_mask)


        p4cfile.write("TIMESTEP  CONTACTS\n")
        p4cfile.write(f"{t} {n_active}\n")
        p4cfile.write("P1  P2  CX  CY  CZ  FX  FY  FZ\n")

        if n_active > 0:

            active_i = np_i[active_mask]
            active_j = np_j[active_mask]

            contact_data = np.column_stack([
                np_ID[active_i],  # P1
                np_ID[active_j],  # P2
                np_position[active_mask, 0],  # CX
                np_position[active_mask, 1],  # CY
                np_position[active_mask, 2],  # CZ
                np_force_a[active_mask, 0],  # FX
                np_force_a[active_mask, 1],  # FY
                np_force_a[active_mask, 2]  # FZ
            ])
            np.savetxt(p4cfile, contact_data, fmt='%d %d %.6e %.6e %.6e %.6e %.6e %.6e')

        tk2 = time.time()
        print(f"save time cost = {tk2 - tk1:.3f}s")

    def init_particle_fields(self, file_name: str, domain_min: Vector3, domain_max: Vector3):

        fp = open(file_name, encoding="UTF-8")
        line: str = fp.readline()  # "TIMESTEP  PARTICLES" line
        line = fp.readline().removesuffix('\n')  # "0 18112" line
        n = int(line.split(' ')[1])

        nwall = 6  # six walls to form a rectangular boundary

        # Initialize particles
        self.gf = Grain.field(shape=(n))
        self.wf = Wall.field(shape=nwall)
        self.wcf = Contact.field(shape=(n, nwall))

        self.mf = Material.field(shape=2)

        line = fp.readline()  # "ID  GROUP  VOLUME  MASS  PX  PY  PZ  VX  VY  VZ" line
        # Processing particles
        max_radius = 0.0
        np_ID = np.zeros(n, int)
        np_groupID = np.zeros(n, int)
        np_density = np.zeros(n, float)
        np_mass = np.zeros(n, float)
        np_radius = np.zeros(n, float)
        np_position = np.zeros((n, 3))
        np_velocity = np.zeros((n, 3))
        np_inertia = np.zeros((n, 3, 3))

        # Extract density, hard coding
        material_density: float = 0.0
        for _ in range(n):
            line = fp.readline()
            if (line == ''): break
            tokens: list[str] = line.split(' ')
            id: int = int(tokens[0])
            i = id - 1
            gid = int(tokens[1])
            radius: float = float(tokens[2])
            mass: float = float(tokens[3])
            px: float = float(tokens[4])
            py: float = float(tokens[5])
            pz: float = float(tokens[6])
            vx: float = float(tokens[7])
            vy: float = float(tokens[8])
            vz: float = float(tokens[9])

            # Hard coding
            density: float = 3 * mass / ( 4 * math.pi * radius ** 3)
            material_density = density

            inertia: float = 2.0 / 5.0 * mass * radius * radius
            np_ID[i] = id
            np_groupID[i] = gid
            # self.gf[i].density = density
            np_density[i] = density
            # self.gf[i].mass = mass
            np_mass[i] = mass
            # self.gf[i].radius = radius
            np_radius[i] = radius
            if (radius > max_radius): max_radius = radius
            # self.gf[i].position = Vector3(px, py, pz)
            np_position[i] = Vector3(px, py, pz)
            # self.gf[i].velocity = Vector3(vx, vy, vz)
            np_velocity[i] = Vector3(vx, vy, vz)
            # self.gf[i].inertia = inertia * ti.Matrix.diag(3, 1.0)
            np_inertia[i] = inertia * ti.Matrix.diag(3, 1.0)
        fp.close()


        self.gf.ID.from_numpy(np_ID)
        self.gf.groupID.from_numpy(np_groupID)
        self.gf.materialType.fill(0)  # Denver Pilphis: hard coding
        self.gf.radius.from_numpy(np_radius)
        self.gf.mass.from_numpy(np_mass)
        self.gf.position.from_numpy(np_position)
        self.gf.velocity.from_numpy(np_velocity)
        self.gf.acceleration.fill(Vector3(0.0, 0.0, 0.0))
        self.gf.force.fill(Vector3(0.0, 0.0, 0.0))
        self.gf.quaternion.fill(Vector4(1.0, 0.0, 0.0, 0.0))
        self.gf.omega.fill((0.0, 0.0, 0.0))
        self.gf.omega_dot.fill(Vector3(0.0, 0.0, 0.0))
        self.gf.inertia.from_numpy(np_inertia)
        self.gf.moment.fill(Vector3(0.0, 0.0, 0.0))

        # Denver Pilphis: hard coding - need to be modified in the future
        for j in range(self.wf.shape[0]):
            self.wf[j].normal = self.config.wall_normals[j]  # Outer normal vector of the wall, [A, B, C]
            self.wf[j].distance = self.config.wall_distances[j] # Distance between origin and the wall, D
            # Material property
            self.wf[j].materialType = 1;  # Hard coding

        contact_model = self.config.contact_model
        # ========================================
        # Material Properties Assignment
        # ========================================

        # --- Particle Material (index 0) ---
        self.mf[0].density = material_density  # Use density from file
        self.mf[0].elasticModulus = self.config.particle_props.elastic_modulus
        self.mf[0].poissonRatio = self.config.particle_props.poisson_ratio

        # --- Wall Material (index 1) ---
        self.mf[1].density = self.config.wall_props.density
        self.mf[1].elasticModulus = self.config.wall_props.elastic_modulus
        self.mf[1].poissonRatio = self.config.wall_props.poisson_ratio

        # ========================================
        # Contact Model Specific Parameters
        # ========================================



        if isinstance(contact_model, LinearContactConfig):
            # ===== Linear Contact Model =====
            print("Setting up Linear contact model parameters...")

            # Stiffness parameters
            self.mf[0].stiffness_normal = contact_model.stiffness_normal
            self.mf[0].stiffness_tangent = contact_model.stiffness_tangential

            # Damping parameters
            self.mf[0].dp_nratio = contact_model.damping_normal
            self.mf[0].dp_tratio = contact_model.damping_tangential

            # Friction coefficients
            self.mf[0].coefficientFriction = contact_model.pp_friction  # Particle-Particle
            self.mf[1].coefficientFriction = contact_model.pw_friction  # Particle-Wall

            # Restitution coefficients (if available)
            self.mf[0].coefficientRestitution = getattr(contact_model, 'pp_restitution', 0.0)
            self.mf[1].coefficientRestitution = getattr(contact_model, 'pw_restitution', 0.0)

        elif isinstance(contact_model, HertzContactConfig):
            # ===== Hertz-Mindlin Contact Model =====
            print("Setting up Hertz-Mindlin contact model parameters...")

            # Hertz model uses restitution coefficients
            self.mf[0].coefficientRestitution = contact_model.pp_restitution  # Particle-Particle
            self.mf[1].coefficientRestitution = contact_model.pw_restitution  # Particle-Wall

            # Friction coefficients
            self.mf[0].coefficientFriction = contact_model.pp_friction
            self.mf[1].coefficientFriction = contact_model.pw_friction

            # Hertz model does NOT use explicit stiffness - calculated from material properties
            # The contact model itself will compute effective stiffness from:
            # - Elastic modulus (already set above)
            # - Poisson ratio (already set above)
            # - Contact geometry (particle radii)

            # Clear any stiffness fields if they exist (avoid confusion)
            if hasattr(self.mf[0], 'stiffness_normal'):
                self.mf[0].stiffness_normal = 0.0
                self.mf[0].stiffness_tangent = 0.0
            if hasattr(self.mf[0], 'dp_nratio'):
                self.mf[0].dp_nratio = 0.0
                self.mf[0].dp_tratio = 0.0

        else:
            raise ValueError(
                f"Unknown contact model type: {type(contact_model).__name__}. "
                f"Expected LinearContactConfig or HertzContactConfig."
            )

        # ========================================
        # Setup Collision Detection
        # ========================================
        self.bpcd = BPCD.create(n, max_radius, domain_min, domain_max)

        # Contact pair bit array
        u1 = ti.types.quant.int(1, False)
        self.cp = ti.field(u1)
        self.cn = ti.root.dense(ti.i, round32(n * n) // 32).quant_array(
            ti.i, dimensions=32, max_num_bits=32).place(self.cp)

        # Contact field
        max_contacts = self.config.particle_props.max_coordinate_number * n
        self.cf = Contact.field(shape=max_contacts)
        self.cfn = ti.field(int, shape=n)

        print(f"Initialized {n} particles with max radius {max_radius:.6f} m")
        print(f"Contact model: {type(contact_model).__name__}")

    def set_contact_model(self, model_type: str = "hertz"):
        """
        Set the contact model type for force calculations.

        Args:
            model_type (str): Contact model type, either "hertz" or "linear".
                             Must match the contact model type in config.

        Raises:
            ValueError: If model_type doesn't match config.contact_model type
        """


        # Validate that requested model matches config
        if model_type.lower() == "hertz":
            if not isinstance(self.config.contact_model, HertzContactConfig):
                raise ValueError(
                    f"Requested 'hertz' model but config uses {type(self.config.contact_model).__name__}. "
                    f"Please use HertzContactConfig in your configuration."
                )
            self.contact_model = HertzMindlinContactModel(self.mf)
            print("✓ Activated Hertz-Mindlin contact model")

        elif model_type.lower() == "linear":
            if not isinstance(self.config.contact_model, LinearContactConfig):
                raise ValueError(
                    f"Requested 'linear' model but config uses {type(self.config.contact_model).__name__}. "
                    f"Please use LinearContactConfig in your configuration."
                )
            self.contact_model = LinearContactModel(self.mf)
            print("✓ Activated Linear contact model")

        else:
            raise ValueError(
                f"Unknown contact model type: '{model_type}'. "
                f"Valid options are 'hertz' or 'linear'."
            )


    # >>> contact field utils
    ###------------------###
    @ti.func
    def append_contact_offset(self, i):
        ret = -1
        offset = ti.atomic_add(self.cfn[i], 1)
        # print(f'i={i}, offset={offset}')
        if offset < self.config.particle_props.max_coordinate_number:
            ret = i * self.config.particle_props.max_coordinate_number + offset
        return ret

    @ti.func
    def search_active_contact_offset(self, i, j):
        ret = -1
        for offset in range(self.cfn[i]):
            if (self.cf[i * self.config.particle_props.max_coordinate_number + offset].j == j
                    and self.cf[i * self.config.particle_props.max_coordinate_number + offset].isActive):
                ret = i * self.config.particle_props.max_coordinate_number + offset
                break
        return ret

    @ti.func
    def remove_inactive_contact(self, i):
        active_count = 0
        for j in range(self.cfn[i]):
            if self.cf[i * self.config.particle_props.max_coordinate_number + j].isActive:
                active_count += 1
        offset = 0
        for j in range(self.cfn[i]):
            if self.cf[i * self.config.particle_props.max_coordinate_number + j].isActive:
                self.cf[i * self.config.particle_props.max_coordinate_number + offset] = self.cf[i * self.config.particle_props.max_coordinate_number + j]
                offset += 1
                if offset >= active_count:
                    break
        for j in range(active_count, self.cfn[i]):
            self.cf[i * self.config.particle_props.max_coordinate_number + j].isActive = False
        self.cfn[i] = active_count

    # <<< contact field utils
    ###------------------###


    # >>> collision bit table utils
    ###------------------###
    @ti.kernel
    def clear_cp_bit_table(self):
        ti.loop_config(bit_vectorize=True)
        for i in ti.grouped(self.cp):
            self.cp[i] = 0

    @ti.func
    def set_collision_bit_callback(self, i: ti.i32, j: ti.i32):
        n = self.gf.shape[0]
        idx = i * n + j
        self.cp[idx] = 1

    @ti.func
    def get_collision_bit(self, i: ti.i32, j: ti.i32):
        n = self.gf.shape[0]
        idx = i * n + j
        return self.cp[idx]

    @ti.kernel
    def cp_bit_table_resolve_collision(self):
        size = self.gf.shape[0]
        for i, j in ti.ndrange(size, size):
            if self.get_collision_bit(i, j):
                self.resolve(i, j)

    # >>> collision bit table utils
    ###------------------###

    @ti.kernel
    def clear_state(self):
        # alias
        gf = ti.static(self.gf)

        for i in gf:
            gf[i].force = Vector3(0.0, 0.0, 0.0)
            gf[i].moment = Vector3(0.0, 0.0, 0.0)

    @ti.kernel
    def late_clear_state(self):
        gf = ti.static(self.gf)
        # remove inactive contact and do compress
        for i in gf:
            self.remove_inactive_contact(i)
    @ti.kernel
    def apply_body_force(self):
        # alias
        # Gravity
        gf = ti.static(self.gf)
        g = self.config.gravity
        for i in gf:
            gf[i].force += gf[i].mass * g
            gf[i].moment += Vector3(0.0, 0.0, 0.0)

            gf[i].force += gf[i].force_fluid
            gf[i].moment += gf[i].moment_fluid


    # NVE integrator with particle state control and improved Velocity Verlet
    @ti.kernel
    def update(self):
        # alias
        gf = ti.static(self.gf)
        mf = ti.static(self.mf)
        dt = self.config.dt

        #kinematic_energy : float = 0.0;

        for i in gf:
            # Check particle state
            if gf[i].freeze:
                # Frozen particle: skip all updates
                continue
            elif gf[i].fixvel:
                # Fixed velocity particle: update position with constant velocity
                gf[i].position += gf[i].velocity * dt
                # Update orientation with constant angular velocity
                # Use quaternion differential equation with fixed angular velocity
                # Eqs. (5)-(16)
                rotational_matrix = quat2RotMatrix(gf[i].quaternion)
                moment_local = rotational_matrix @ gf[i].moment
                omega_local = rotational_matrix @ gf[i].omega
                omega_dot_local = ti.Matrix.inverse(gf[i].inertia) @ (
                        moment_local - omega_local.cross(gf[i].inertia @ omega_local))
                alpha = ti.Matrix.inverse(rotational_matrix) @ omega_dot_local

                # Update angular velocity using average of previous and current angular acceleration
                gf[i].omega += 0.5 * (gf[i].omega_dot + alpha) * dt

                # Update particle orientation using quaternion differential equation
                # Reference: Lu et al. (2015) Discrete element models for non-spherical particle systems: From theoretical developments to applications.
                # http://dx.doi.org/10.1016/j.ces.2014.11.050
                # Eq. (6)
                # Originally from Langston et al. (2004) Distinct element modelling of non-spherical frictionless particle flow.
                # https://doi.org/10.1016/j.ces.2003.10.008
                dq0 = - 0.5 * (
                        gf[i].quaternion[1] * gf[i].omega[0] + gf[i].quaternion[2] * gf[i].omega[1] + gf[i].quaternion[
                    3] * gf[i].omega[2])
                dq1 = + 0.5 * (
                        gf[i].quaternion[0] * gf[i].omega[0] - gf[i].quaternion[3] * gf[i].omega[1] + gf[i].quaternion[
                    2] * gf[i].omega[2])
                dq2 = + 0.5 * (
                        gf[i].quaternion[3] * gf[i].omega[0] + gf[i].quaternion[0] * gf[i].omega[1] + gf[i].quaternion[
                    1] * gf[i].omega[2])
                dq3 = + 0.5 * (
                        -gf[i].quaternion[2] * gf[i].omega[0] + gf[i].quaternion[1] * gf[i].omega[1] + gf[i].quaternion[
                    0] * gf[i].omega[2])

                gf[i].quaternion[0] += dq0 * dt
                gf[i].quaternion[1] += dq1 * dt
                gf[i].quaternion[2] += dq2 * dt
                gf[i].quaternion[3] += dq3 * dt
                gf[i].quaternion = tm.normalize(gf[i].quaternion)

                # Store current angular acceleration for next time step
                gf[i].omega_dot = alpha

                # Skip force and acceleration calculations for fixed velocity particles
                continue
            else:
                # Normal particle: full dynamics update
                # Translational motion
                # Improved Velocity Verlet integrator using average of current and previous acceleration
                # Calculate current acceleration
                a = gf[i].force / gf[i].mass
                # Update velocity using average of previous and current acceleration
                gf[i].velocity += 0.5 * (gf[i].acceleration + a) * dt

                # Update position using current velocity and acceleration
                gf[i].position += gf[i].velocity * dt + 0.5 * a * dt ** 2

                # Store current acceleration for next time step
                gf[i].acceleration = a

                # Rotational motion
                # Calculate angular acceleration using Euler's equation for rigid body
                # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
                # https://doi.org/10.1002/nme.6568
                # Eqs. (5)-(16)
                rotational_matrix = quat2RotMatrix(gf[i].quaternion)
                moment_local = rotational_matrix @ gf[i].moment
                omega_local = rotational_matrix @ gf[i].omega
                omega_dot_local = ti.Matrix.inverse(gf[i].inertia) @ (
                        moment_local - omega_local.cross(gf[i].inertia @ omega_local))
                alpha = ti.Matrix.inverse(rotational_matrix) @ omega_dot_local

                # Update angular velocity using average of previous and current angular acceleration
                gf[i].omega += 0.5 * (gf[i].omega_dot + alpha) * dt

                # Update particle orientation using quaternion differential equation
                # Reference: Lu et al. (2015) Discrete element models for non-spherical particle systems: From theoretical developments to applications.
                # http://dx.doi.org/10.1016/j.ces.2014.11.050
                # Eq. (6)
                # Originally from Langston et al. (2004) Distinct element modelling of non-spherical frictionless particle flow.
                # https://doi.org/10.1016/j.ces.2003.10.008
                dq0 = - 0.5 * (
                        gf[i].quaternion[1] * gf[i].omega[0] + gf[i].quaternion[2] * gf[i].omega[1] + gf[i].quaternion[
                    3] * gf[i].omega[2])
                dq1 = + 0.5 * (
                        gf[i].quaternion[0] * gf[i].omega[0] - gf[i].quaternion[3] * gf[i].omega[1] + gf[i].quaternion[
                    2] * gf[i].omega[2])
                dq2 = + 0.5 * (
                        gf[i].quaternion[3] * gf[i].omega[0] + gf[i].quaternion[0] * gf[i].omega[1] + gf[i].quaternion[
                    1] * gf[i].omega[2])
                dq3 = + 0.5 * (
                        -gf[i].quaternion[2] * gf[i].omega[0] + gf[i].quaternion[1] * gf[i].omega[1] + gf[i].quaternion[
                    0] * gf[i].omega[2])

                gf[i].quaternion[0] += dq0 * dt
                gf[i].quaternion[1] += dq1 * dt
                gf[i].quaternion[2] += dq2 * dt
                gf[i].quaternion[3] += dq3 * dt
                gf[i].quaternion = tm.normalize(gf[i].quaternion)

                # Store current angular acceleration for next time step
                gf[i].omega_dot = alpha
            #ti.atomic_add(kinematic_energy, gf[i].mass / 2.0 * tm.dot(gf[i].velocity, gf[i].velocity)
                          #+ gf[i].mass * gf[i].radius/ 5.0 * tm.dot(gf[i].omega, gf[i].omega));

        #print(f"{kinematic_energy}");

    @ti.func
    def resolve(self, i: int, j: int):
        '''
        Particle-particle contact detection
        '''
        # alias
        gf = ti.static(self.gf)
        cf = ti.static(self.cf)

        mf = ti.static(self.mf)

        eval = False
        # Particle-particle contacts
        offset = self.search_active_contact_offset(i, j)

        if offset >= 0:  # Existing contact
            # Check if particles are still in contact
            gap = tm.length(gf[j].position - gf[i].position) - gf[i].radius - gf[j].radius
            if gap < 0:  # Still in contact
                eval = True
            else:
                cf[offset].isActive = 0
        else:
            # Check for new contact
            gap = tm.length(gf[j].position - gf[i].position) - gf[i].radius - gf[j].radius
            if gap < 0:  # New contact detected
                offset = self.append_contact_offset(i)
                if offset < 0:
                    print(
                        f"ERROR: coordinate number > set_max_coordinate_number({self.config.particle_props.max_coordinate_number})")
                cf[offset] = Contact(
                    i=i,
                    j=j,
                    isActive=1,
                    materialType_i=0,
                    materialType_j=0,
                    force_a=Vector3(0.0, 0.0, 0.0),
                    moment_a=Vector3(0.0, 0.0, 0.0),
                    moment_b=Vector3(0.0, 0.0, 0.0),
                    shear_displacement=Vector3(0.0, 0.0, 0.0)
                )
                eval = True  # Send to evaluation using  contact model

        if (eval):
            dt = self.config.dt
            # Contact resolution
            # Find out rotation matrix
            # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
            a = tm.normalize(gf[j].position - gf[i].position)
            b = Vector3(1.0, 0.0, 0.0)  # Local x coordinate
            v = tm.cross(a, b)
            s = tm.length(v)
            c = tm.dot(a, b)
            rotationMatrix = Zero3x3();
            if (s < DoublePrecisionTolerance):
                if (c > 0.0):
                    rotationMatrix = ti.Matrix.diag(3, 1.0)
                else:
                    rotationMatrix = DEMMatrix([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
            else:
                vx = DEMMatrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
                rotationMatrix = ti.Matrix.diag(3, 1.0) + vx + ((1.0 - c) / s ** 2) * vx @ vx

            length = tm.length(gf[j].position - gf[i].position)

            gap = length - gf[i].radius - gf[j].radius  # gap must be negative to ensure an intact contact
            delta_n = -1 * gap  # For parameter calculation only

            # For debug only
            if delta_n > 0.05 * ti.min(gf[i].radius, gf[j].radius):
                print("WARNING: Overlap particle-particle exceeds 0.05")

            cf[offset].position = gf[i].position + tm.normalize(gf[j].position - gf[i].position) * (
                    gf[i].radius - delta_n)
            r_i = cf[offset].position - gf[i].position
            r_j = cf[offset].position - gf[j].position

            # Velocity of a point on the surface of a rigid body
            v_c_i = tm.cross(gf[i].omega, r_i) + gf[i].velocity
            v_c_j = tm.cross(gf[j].omega, r_j) + gf[j].velocity
            v_c = rotationMatrix @ (v_c_j - v_c_i)  # LOCAL coordinate

            # Parameter calculation

            # Contact model
            F = self.contact_model.particle_particle_force(i, j, gf, cf, offset, dt ,delta_n, v_c)

            # For P4C output
            cf[offset].force_a = F
            # cf[offset].force_b = -F;
            # Assigning contact force to particles
            # Notice the inverse of signs due to Newton's third law
            # and LOCAL to GLOBAL coordinates
            F_i_global = ti.Matrix.inverse(rotationMatrix) @ (-F)
            F_j_global = ti.Matrix.inverse(rotationMatrix) @ F
            ti.atomic_add(gf[i].force, F_i_global)
            ti.atomic_add(gf[j].force, F_j_global)
            # As the force is at contact position
            # additional moments will be assigned to particles
            # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
            # https://doi.org/10.1002/nme.6568
            # Eqs. (3)-(4)
            ti.atomic_add(gf[i].moment, tm.cross(r_i, F_i_global))
            ti.atomic_add(gf[j].moment, tm.cross(r_j, F_j_global))

    @ti.func
    def evaluate_wall(self, i: int, j: int):  # i is particle, j is wall
        '''
        # i is particle, j is wall
        Particle-wall contact evaluation
        Contact model is Hertz-Mindlin
        '''

        # alias
        gf = ti.static(self.gf)
        wf = ti.static(self.wf)
        wcf = ti.static(self.wcf)


        dt = self.config.dt
        # Contact resolution
        # Find out rotation matrix
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        a = wf[j].normal
        b = Vector3(1.0, 0.0, 0.0)  # Local x coordinate
        v = tm.cross(a, b)
        s = tm.length(v)
        c = tm.dot(a, b)
        rotationMatrix = Zero3x3()
        if (s < DoublePrecisionTolerance):
            if (c > 0.0):
                rotationMatrix = ti.Matrix.diag(3, 1.0);
            else:
                rotationMatrix = DEMMatrix([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        else:
            vx = DEMMatrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
            rotationMatrix = ti.Matrix.diag(3, 1.0) + vx + ((1.0 - c) / s ** 2) * vx @ vx

        # Calculation relative translational and rotational displacements
        delta_n = (tm.dot(gf[i].position, wf[j].normal) - wf[j].distance) + gf[i].radius
        gap = -1 * delta_n          # gap must be negative
        distance = gf[i].radius - delta_n

        # For debug only
        if delta_n > 0.05 * gf[i].radius:
            print("WARNING: Overlap particle-wall exceeds 0.05")

        r_i =  distance * wf[j].normal
        wcf[i, j].position = gf[i].position + r_i
        # Velocity of a point on the surface of a rigid body
        v_c_i = tm.cross(gf[i].omega, r_i) + gf[i].velocity
        v_c = rotationMatrix @ (- v_c_i)  # LOCAL coordinate
        # Parameter calculation
        # Contact model
        F = self.contact_model.particle_wall_force(i, j, gf, wf, wcf, dt, delta_n, v_c)
        # For P4C output
        wcf[i, j].force_a = F
        #print(f"force_a {F[0]}{F[1]}")
        # wcf[i, j].force_b = -F;
        # Assigning contact force to particles
        # Notice the inverse of signs due to Newton's third law
        # and LOCAL to GLOBAL coordinates
        # As the force is at contact position
        # additional moments will be assigned to particles
        F_i_global = ti.Matrix.inverse(rotationMatrix) @ (-F)

        ti.atomic_add(gf[i].force, F_i_global)
        # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
        # https://doi.org/10.1002/nme.6568
        # Eqs. (3)-(4)
        ti.atomic_add(gf[i].moment, tm.cross(r_i, F_i_global))

    @ti.kernel
    def resolve_wall(self):
        '''
        Particle-wall contact detection
        '''

        # alias
        gf = ti.static(self.gf)
        wf = ti.static(self.wf)
        wcf = ti.static(self.wcf)
        # All particles will be contact detection with the wall
        for i, j in ti.ndrange(gf.shape[0], wf.shape[0]):
            # Particle-wall contacts
            if wcf[i, j].isActive:  # Existing contact
                if -(tm.dot(gf[i].position, wf[j].normal) - wf[j].distance) > gf[i].radius:  # Non-contact
                    wcf[i, j].isActive = 0
                else:  # Contact
                    self.evaluate_wall(i, j)
            else:
                if -(tm.dot(gf[i].position, wf[j].normal) - wf[j].distance) <= gf[i].radius:  # Contact
                    wcf[i, j] = Contact(
                        isActive=1,
                        materialType_i=0,
                        materialType_j=1,
                        shear_displacement=Vector3(0.0, 0.0, 0.0)
                    )
                    self.evaluate_wall(i, j)

    def contact(self):
        '''
        Handle the collision between grains.
        '''
        self.clear_cp_bit_table()
        self.bpcd.detect_collision(self.gf.position, self.set_collision_bit_callback)
        self.cp_bit_table_resolve_collision()

    def run_simulation(self):
        '''
        Run one step of the simulation
        '''
        self.clear_state()
        # Particle-particle collision
        self.contact()
        # Particle-wall collision
        self.resolve_wall()
        # Particle body force
        self.apply_body_force()
        # Time integration
        self.update()
        # Clear certain states at the end
        self.late_clear_state()