"""
Hertz-Mindlin contact force model for DEM simulations.

Implements normal and tangential force computation for both particle–particle
and particle–wall interactions based on the Hertz-Mindlin theory with
viscoelastic damping and Coulomb friction. Material properties are derived
from Young’s modulus, Poisson’s ratio, restitution, and friction coefficients.
"""

import math
import taichi.math as tm
import taichi as ti
from .contactmodel import ContactModel

#=====================================
# Type Definition
#=====================================

Vector3 = ti.types.vector(3, float)

@ti.data_oriented
class HertzMindlinContactModel(ContactModel):
    """
        Hertz-Mindlin contact model for discrete element method simulation.

        This class implements particle-particle and particle-wall contact force calculations
        based on the Hertz-Mindlin contact model.

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
        Y_star = 1.0 / ((1.0 - mf[type_i].poissonRatio ** 2) / mf[type_i].elasticModulus + (
                1.0 - mf[type_j].poissonRatio ** 2) / mf[type_j].elasticModulus)
        G_star = 1.0 / (2.0 * (2.0 - mf[type_i].poissonRatio) * (1.0 + mf[type_i].poissonRatio) / mf[
            type_i].elasticModulus + 2.0 * (2.0 - mf[type_j].poissonRatio) * (1.0 + mf[type_j].poissonRatio) /
                        mf[type_j].elasticModulus)
        R_star = 1.0 / (1.0 / gf[i].radius + 1.0 / gf[j].radius)
        rho = gf[i].mass * 3 / (4 * tm.pi* gf[i].radius ** 3)
        t = self.min_time(R_star,rho , G_star , mf[type_i].poissonRatio)
        if dt > t:
            print(f"Time Warning!  {dt:.6f}> {t:.8f}")
        m_star = 1.0 / (1.0 / gf[i].mass + 1.0 / gf[j].mass)
        # Restitution and friction coefficients
        e = 0.5 * (mf[type_i].coefficientRestitution + mf[type_j].coefficientRestitution)
        mu = 0.5 * (mf[type_i].coefficientFriction + mf[type_j].coefficientFriction)
        beta = tm.log(e) / tm.sqrt(tm.log(e) ** 2 + math.pi ** 2)
        S_n = 2.0 * Y_star * tm.sqrt(R_star * delta_n)
        S_t = 8.0 * G_star * tm.sqrt(R_star * delta_n)
        k_n = 4.0 / 3.0 * Y_star * tm.sqrt(R_star * delta_n)
        gamma_n = - 2.0 * beta * tm.sqrt(5.0 / 6.0 * S_n * m_star)  # Check whether gamma_n >= 0
        k_t = 8.0 * G_star * tm.sqrt(R_star * delta_n)
        gamma_t = - 2.0 * beta * tm.sqrt(5.0 / 6.0 * S_t * m_star)  # Check whether gamma_t >= 0

        # Shear displacement increments
        shear_increment = v_c * dt
        shear_increment[0] = 0.0  # Remove the normal direction
        cf[offset].shear_displacement += shear_increment
        # Normal direction - LOCAL - the force towards particle j
        F = Vector3(0.0, 0.0, 0.0)

        # Be aware of signs
        F[0] = - k_n * gap - gamma_n * v_c[0]
        # Shear direction - LOCAL - the force towards particle j

        try_shear_force = - k_t * cf[offset].shear_displacement
        if tm.length(try_shear_force) >= mu * F[0]:  # Sliding
            ratio: ti.f64 = mu * F[0] / tm.length(try_shear_force)
            F[1] = try_shear_force[1] * ratio
            F[2] = try_shear_force[2] * ratio
            cf[offset].shear_displacement[1] = -F[1] / k_t
            cf[offset].shear_displacement[2] = -F[2] / k_t
        else:  # No sliding
            F[1] = try_shear_force[1] - gamma_t * v_c[1]
            F[2] = try_shear_force[2] - gamma_t * v_c[2]


        return F

    @ti.func
    def particle_wall_force(self, i, j, gf, wf, wcf, dt, delta_n, v_c:Vector3)-> Vector3:

        mf = ti.static(self.mf)
        # Reference: https://www.cfdem.com/media/DEM/docu/gran_model_hertz.html
        type_i: ti.i32 = gf[i].materialType
        type_j: ti.i32 = wf[j].materialType

        gap = -1 * delta_n
        Y_star = 1.0 / ((1.0 - mf[type_i].poissonRatio ** 2) / mf[type_i].elasticModulus +
         (1.0 - mf[type_j].poissonRatio ** 2) / mf[type_j].elasticModulus )
        G_star = 1.0 / (2.0 * (2.0 - mf[type_i].poissonRatio) * (1.0 + mf[type_i].poissonRatio) / mf[
            type_i].elasticModulus +  2.0 * (2.0 - mf[type_j].poissonRatio) * (1.0 + mf[type_j].poissonRatio) / mf[
                    type_j].elasticModulus )
        R_star = gf[i].radius
        m_star = gf[i].mass
        # Restitution and friction coefficients
        e = mf[type_j].coefficientRestitution
        mu = mf[type_j].coefficientFriction

        log_e = ti.log(e)
        beta = log_e / ti.sqrt(log_e ** 2 + math.pi ** 2)
        S_n = 2.0 * Y_star * tm.sqrt(R_star * delta_n)
        S_t = 8.0 * G_star * tm.sqrt(R_star * delta_n)
        k_n = 4.0 / 3.0 * Y_star * tm.sqrt(R_star * delta_n)
        gamma_n = - 2.0 * beta * tm.sqrt(5.0 / 6.0 * S_n * m_star)  # Check whether gamma_n >= 0
        k_t = 8.0 * G_star * tm.sqrt(R_star * delta_n)
        gamma_t = - 2.0 * beta * tm.sqrt(5.0 / 6.0 * S_t * m_star)  # Check whether gamma_t >= 0

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
        F[0] = - k_n * gap - gamma_n * v_c[0]
        # Shear direction - LOCAL - the force towards the wall

        try_shear_force  = - k_t * wcf[i, j].shear_displacement
        if tm.length(try_shear_force) >= mu * F[0]:  # Sliding
            ratio = mu * F[0] / tm.length(try_shear_force)
            F[1] = try_shear_force[1] * ratio
            F[2] = try_shear_force[2] * ratio
            wcf[i, j].shear_displacement[1] = -F[1] / k_t
            wcf[i, j].shear_displacement[2] = -F[2] / k_t
            #print(f"f {F[1]} F { F[0]}")
        else:  # No sliding
            F[1] = try_shear_force[1] - gamma_t * v_c[1]
            F[2] = try_shear_force[2] - gamma_t * v_c[2]


        return F


    @ti.func
    def min_time(self ,r:float, rho:float,Gm:float,v:float):
        a =  0.1631 * v + 0.8766
        b = rho / Gm
        t = tm.pi * r * ti.sqrt(b) / a

        return  t



