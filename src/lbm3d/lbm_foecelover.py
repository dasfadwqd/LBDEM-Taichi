'''
A module that contains the collision operator with the body force.

'''

import taichi as ti
from lbm_solver3d import BasicLattice3D
from lbmutils import CellType

# =====================================
# Type Definitions
# =====================================
Vector3 = ti.types.vector(3, float)

@ti.data_oriented
class ForcedLattice3D(BasicLattice3D):
    def __init__(self, Nx: int, Ny: int, Nz: int, omega: float, dx: float, dt: float, rho: float , F :Vector3):
        """Generates a lattice with mesoscopic and macroscopic variables for BGK collision operator with a body force.

        Args:
            Nx (int): Number of lattice nodes in x-direction.
            Ny (int): Number of lattice nodes in y-direction.
            Nz (int): Number of lattice nodes in z-direction.
            omega (float): Relaxation frequency.
            dx (float): Lattice spacing [m].
            dt (float): LBM time step [s].
            rho (float): Fluid densty [kg/m^3].
             F (Vector3): Body force acceleration [m/s^2].
        """
        super().__init__(Nx, Ny, Nz ,omega, dx, dt, rho)            # inheritance from BasicLattice3D
        self.F = self.unit.getLbAccel(F)                        # body force


    #==============================#
    # ----- Collision Process -----#
    #==============================#
    @ti.kernel

    def collide(self):
        """BGK collision operator in LBM.
                fpc = (1-omega)*f + omega*feq
        """
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # skip the boundary nodes
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            # update the equilibrium state
            self.compute_feq(i, j, k)

            # collision (relax the distribution functions towards equilibrium)
            for q in ti.static(range(BasicLattice3D.Q)):
                #  c[q] · F
                cdotF = (ForcedLattice3D.c[q].x * self.F.x +
                         ForcedLattice3D.c[q].y * self.F.y +
                         ForcedLattice3D.c[q].z * self.F.z)

                # calculate the force term
                Fq = (self.rho[i, j, k] *
                      ForcedLattice3D.w[q] * cdotF /
                      ForcedLattice3D.cssq)

                # relax the distribution functions
                self.fpc[i, j, k][q] = (
                        (1.0 - self.omega[i, j, k]) * self.f[i, j, k][q] +
                        self.omega[i, j, k] * self.feq[i, j, k][q] +
                        Fq
                )
