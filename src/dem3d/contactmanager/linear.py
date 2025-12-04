"""
Linear spring-dashpot contact model for DEM simulations.

Implements normal and tangential forces using constant stiffness and
viscous damping coefficients. Supports particle–particle and particle–wall
interactions with Coulomb friction and history-dependent shear displacement.
"""
import taichi.math as tm
import taichi as ti
from .contactmodel import ContactModel

#=====================================
# Type Definition
#=====================================

Vector3 = ti.types.vector(3, float)


@ti.data_oriented
class LinearContactModel(ContactModel):
    """
    Linear contact model for discrete element method simulation.

    This class implements particle-particle and particle-wall contact force calculations
    based on the Linear contact model.

    The model accounts for:
    - Elastic deformation
    - Damping
    - Coulomb friction
    - Particle and wall material properties
    """

    def __init__(self, material_field):
        """
        Initialize the Hertz-Mindlin contact model.

        Args:
            material_field (MaterialField): Field containing material properties for particles
        """
        super().__init__(material_field)

    @ti.func
    def particle_particle_force(self, i, j, gf, cf, offset, dt, delta_n, v_c:Vector3)-> Vector3:

        mf = ti.static(self.mf)
        gap = -1 * delta_n
        type_i: ti.i32 = gf[i].materialType
        type_j: ti.i32 = gf[j].materialType
        # Harmonic mean of stiffness values
        kn_star = mf[type_i].stiffness_normal * mf[type_j].stiffness_normal \
                  / (mf[type_i].stiffness_normal + mf[type_j].stiffness_normal)
        ks_star = mf[type_i].stiffness_tangent * mf[type_j].stiffness_tangent \
                  / (mf[type_i].stiffness_tangent + mf[type_j].stiffness_tangent)
        R_star = 1.0 / (1.0 / gf[i].radius + 1.0 / gf[j].radius)

        # Calculate average damping ratio
        dp_nratio = 0.5 * (mf[type_i].dp_nratio + mf[type_j].dp_nratio)
        dp_sratio = 0.5 * (mf[type_i].dp_sratio + mf[type_j].dp_sratio)


        t = self.min_time(gf[i].mass,mf[type_i].stiffness_normal)
        if dt > t:
            print(f"Time Warning!  {dt:.6f}> {t:.6f}")
        m_star = 1.0 / (1.0 / gf[i].mass + 1.0 / gf[j].mass)
        # Restitution and friction coefficients
        e = 0.5 * (mf[type_i].coefficientRestitution + mf[type_j].coefficientRestitution)
        # Calculate friction coefficient
        mu = 0.5 * (mf[type_i].coefficientFriction + mf[type_j].coefficientFriction)

        # Calculate damping coefficient
        # Uses a formulation based on material damping ratio and effective mass
        Cn = 2. * dp_nratio * ti.sqrt(kn_star * m_star)   # Cn > 0
        Ct = 2. * dp_sratio * ti.sqrt(kn_star * m_star)  # Ct > 0

        # Shear displacement increments
        shear_increment = v_c * dt
        shear_increment[0] = 0.0  # Remove the normal direction
        cf[offset].shear_displacement += shear_increment
        # Normal direction - LOCAL - the force towards particle j
        F = Vector3(0.0, 0.0, 0.0)
        # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
        # https://doi.org/10.1002/nme.6568
        # Eq. (29)
        # Be aware of signs
        F[0] = - kn_star * gap - Cn * v_c[0]
        # Shear direction - LOCAL - the force towards particle j

        try_shear_force = - ks_star * cf[offset].shear_displacement
        if tm.length(try_shear_force) >= mu * F[0]:  # Sliding
            ratio: ti.f64 = mu * F[0] / tm.length(try_shear_force)
            F[1] = try_shear_force[1] * ratio
            F[2] = try_shear_force[2] * ratio
            cf[offset].shear_displacement[1] = -F[1] / ks_star
            cf[offset].shear_displacement[2] = -F[2] / ks_star
        else:  # No sliding
            F[1] = try_shear_force[1] - Ct * v_c[1]
            F[2] = try_shear_force[2] - Ct * v_c[2]


        return F

    @ti.func
    def particle_wall_force(self, i, j, gf, wf, wcf, dt, delta_n, v_c:Vector3)-> Vector3:

        mf = ti.static(self.mf)
        # Reference: https://www.cfdem.com/media/DEM/docu/gran_model_hertz.html
        type_i: ti.i32 = gf[i].materialType
        type_j: ti.i32 = wf[j].materialType

        gap = -1 * delta_n
        # Calculate effective stiffness parameters
        kn_star = mf[type_i].stiffness_normal
        ks_star = mf[type_i].stiffness_tangent
        dp_nratio = mf[type_i].dp_nratio
        dp_sratio = mf[type_i].dp_sratio

        R_star = gf[i].radius
        m_star = gf[i].mass

        mu = mf[type_j].coefficientFriction

        # Calculate damping coefficient
        Cn = 2. * dp_nratio * ti.sqrt(kn_star * m_star)  # Cn > 0
        Ct = 2. * dp_sratio * ti.sqrt(kn_star * m_star)  # Ct > 0

        # Shear displacement increments
        shear_increment = v_c * dt
        shear_increment[0] = 0.0  # Remove the normal direction
        wcf[i, j].shear_displacement += shear_increment
        # Normal direction - LOCAL - the force towards the wall
        F = Vector3(0.0, 0.0, 0.0)
        # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
        # https://doi.org/10.1002/nme.6568
        # Eq. (29)
        # Be aware of signs
        F[0] = - kn_star * gap - Cn * v_c[0]
        # Shear direction - LOCAL - the force towards the wall

        try_shear_force  = - ks_star * wcf[i, j].shear_displacement
        if tm.length(try_shear_force) >= mu * F[0]:  # Sliding
            ratio = mu * F[0] / tm.length(try_shear_force)
            F[1] = try_shear_force[1] * ratio
            F[2] = try_shear_force[2] * ratio
            wcf[i, j].shear_displacement[1] = -F[1] / ks_star
            wcf[i, j].shear_displacement[2] = -F[2] / ks_star
        else:  # No sliding
            F[1] = try_shear_force[1] - Ct * v_c[1]
            F[2] = try_shear_force[2] - Ct * v_c[2]


        return F


    @ti.func
    def min_time(self ,m :float, K :float):

        t = ti.sqrt(m / K)

        return  t



