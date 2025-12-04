'''
A module that contains the utilities for the lattice Boltzmann calculations.
'''

import taichi as ti
# =====================================
# Type Definitions
# =====================================
Vector3 = ti.types.vector(3, float)

# ===========================#
# ----- Convert Units ----- #
# ===========================#
class UnitConverter:
    def __init__(self, dx: float, dt: float, rho: float):
        """Converts between physical and lattice units.

        Args:
            dx (float): Lattice spacing.
            dt (float): Time step.
            rho (float): Fluid density.
        """
        self.dx = dx
        self.dt = dt
        self.rho = rho

    # from the physical units to lattice units
    def getLbLength(self, physLength): return physLength / self.dx

    def getLbTime(self, physTime): return physTime / self.dt

    def getLbVel(self, physVel): return physVel * self.dt / self.dx

    def getLbAccel(self, physAccel): return physAccel * (self.dt ** 2) / self.dx

    def getLbStrainRate(self, physStrainRate): return physStrainRate * self.dt

    def getLbRho(self, physRho): return physRho / self.rho

    def getLbNu(self, physNu): return physNu * self.dt / (self.dx ** 2)

    def getLbSigma(self, physSigma): return physSigma * (self.dt ** 2) / (self.rho * (self.dx ** 2))

    def getLbForce(self, physForce): return physForce * (self.dt ** 2) / (self.rho * (self.dx ** 4))

    # from the lattice units to physical units
    def getPhysLength(self, lbLength): return lbLength * self.dx

    def getPhysTime(self, lbTime): return lbTime * self.dt

    def getPhysVel(self, lbVel): return lbVel * self.dx / self.dt

    def getPhysAccell(self, lbAccel): return lbAccel * self.dx / (self.dt ** 2)

    def getPhysStrainRate(self, lbStrainRate): return lbStrainRate / self.dt

    def getPhysRho(self, lbRho): return lbRho * self.rho

    def getPhysNu(self, lbNu): return lbNu * (self.dx ** 2) / self.dt

    def getPhysSigma(self, lbSigma): return lbSigma * (self.rho * (self.dx ** 2)) / (self.dt ** 2)

    def getPhysForce(self, lbForce): return lbForce * (self.rho * (self.dx ** 4)) / (self.dt ** 2)


# ========================#
# ----- Cell Types ----- #
# ========================#
class CellType:
    # material types
    FLUID = 0b_0000_0000_0000_0001  # 1: fluid cells
    INTERFACE = 0b_0000_0000_0000_0010  # 2: interface cells
    GAS = 0b_0000_0000_0000_0100  # 4: gas cells
    OBSTACLE = 0b_0000_0000_0000_1000  # 8: no-slip bounce-back cells

    # boundary conditions 1
    FREE_SLIP = 0b_0000_0000_0001_0000  # 16: free-slip cells
    VEL_LADD = 0b_0000_0000_0010_0000  # 32: moving boundary cells (both tangential and normal motions may exist!)
    # VEL_EQ =        0b_0000_0000_0100_0000    # 64: boundary cells with a fixed normal velocity (equilibrium method)
    VEL_ZOUHE = 0b_0000_0000_0100_0000  # 64: boundary nodes with a fixed velocity
    VEL_EXIT = 0b_0000_0000_1000_0000  # 128: exit boundary condition with zero velocity gradient


    Pre_ZOUHE = 0b_0000_0001_0000_0000  # 256: pressure boundary
    # boundary conditions 2
    NO_FLUID_NEIGH = 0b_0000_0010_0000_0000  # 512: no gas neighbour interface cells
    TO_FLUID = 0b_0000_0100_0000_0000  # 1024: to fluid interface cells
    TO_GAS = 0b_0000_1000_0000_0000  # 2048: to gas interface cells

    # wall positions
    RIGHT = 0b_0001_0000_0000_0000  # 4096: the right wall
    LEFT = 0b_0010_0000_0000_0000  # 8192: the left wall
    TOP = 0b_0100_0000_0000_0000  # 16384: the top wall
    BOTTOM = 0b_1000_0000_0000_0000  # 32768: the bottom wall
    FRONT = 0b0001_0000_0000_0000_0000  # 65536: the front wall
    BACK = 0b0010_0000_0000_0000_0000  # 131072: the back wall
