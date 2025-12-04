"""
Contact force models for particle–particle and particle–wall interactions in DEM simulations.

This module defines the interface for contact force computation, supporting both
particle–particle and particle–wall scenarios using material properties and relative
kinematics at the contact point.
"""

import taichi as ti


@ti.data_oriented
class ContactModel:

    def __init__(self, material_field):
        self.mf = material_field

    @ti.func
    def particle_particle_force(self, i, j, gf, cf, offset, dt, delta_n, v_c):
        """
        Compute contact force between two particles.

        Args:
            i (int): Index of particle i
            j (int): Index of particle j
            gf (GrainField): Particle data field
            cf (ContactField): Contact data field
            offset (int): Contact index offset
            dt (float): Time step
            delta_n (float): Normal overlap
            v_c (Vector3): Relative velocity in local contact frame

        Returns:
            Contact force in local contact coordinates (Vector3).
        """
        pass

    @ti.func
    def particle_wall_force(self, i, j, gf, wf, wcf, dt, delta_n, v_c):
        """
        Compute contact force between a particle and a wall.

        Args:
            i (int): Particle index
            j (int): Wall index
            gf (GrainField): Particle data field
            wf (WallField): Wall data field
            wcf (WallContactField): Wall contact data field
            dt (float): Time step
            delta_n (float): Normal overlap
            v_c (Vector3): Relative velocity in local contact frame

        Returns:
            Contact force in local contact coordinates (Vector3).
        """
        pass