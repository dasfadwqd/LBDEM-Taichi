'''
The partially saturated cell (immersed moving boundary) method for 3D LBM-DEM coupling.

'''

import taichi as ti
import taichi.math as tm

# LBM module
from src.lbm3d.lbm_solver3d import BasicLattice3D
from src.lbm3d.lbmutils import CellType
# DEM module
from src.dem3d.demsolver import DEMSolver

# =====================================
# Type Definition
# =====================================
Vector3 = ti.types.vector(3,float)



class PSCLattice3D(BasicLattice3D):
    """Couples LBM and DEM via the partially saturated cell method in 3D.

    Args:
        BasicLattice3D (class): Basic 3D lattice class.
    """

    def __init__(self, Nx: int, Ny: int, Nz: int, omega: float, dx: float, dt: float, rho: float, demslover: DEMSolver):
        """Generate a 3D lattice with the mesoscopic and macroscopic variables.

        Args:
            Nx (int): Number of lattice nodes in x-direction.
            Ny (int): Number of lattice nodes in y-direction.  
            Nz (int): Number of lattice nodes in z-direction.
            omega (float): Relaxation frequency.
            dx (float): Lattice spacing [m].
            dt (float): LBM time step [s].
            rho (float): Fluid density [kg/m^3].
            demslover (DEMSolver): DEM solver instance.
        """
        super().__init__(Nx, Ny, Nz, omega, dx, dt, rho)

        # additional lattice properties for FSI coupling
        self.id = ti.field(int, shape=(Nx, Ny, Nz))  # tag cell with grain id
        self.id.fill(-1)  # no grain has id of -1
        self.volfrac = ti.field(float, shape=(Nx, Ny, Nz))  # solid volume fraction
        self.weight = ti.field(float, shape=(Nx, Ny, Nz))  # weighting factor
        self.velsolid = ti.Vector.field(PSCLattice3D.D, float, shape=(Nx, Ny, Nz))  # solid velocity
        self.feqsolid = ti.Vector.field(PSCLattice3D.Q, float, shape=(Nx, Ny, Nz))  # feq based on solid velocity
        self.hydroforce = ti.Vector.field(PSCLattice3D.D, float, shape=(Nx, Ny, Nz))  # hydrodynamic force field
        self.hydrotorque = ti.Vector.field(PSCLattice3D.D, float, shape=(Nx, Ny, Nz))  # hydrodynamic torque field

        # invoke DEM for data exchange
        self.dem = demslover

    # ============================#
    # ----- Initialization ----- #
    # ============================#
    @ti.kernel
    def initialize(self):
        """Initializes the 3D lattice.
        """
        # set density distribution functions to the equilibrium states
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # skip the boundary cells
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            # calculate the equilibrium distribution functions
            self.compute_feq(i, j, k)

            # assign the initial distribution function as the equilibrium state
            for q in ti.static(range(PSCLattice3D.Q)):
                self.f[i, j, k][q] = self.feq[i, j, k][q]

        # map grains to lattice in case the initial state needs to be saved
        self.grains2lattice()

    # ===========================================#
    # ----- Data Exchange from DEM to LBM ----- #
    # ===========================================#

    @ti.func
    def grains2lattice(self):
        """Map grain information (id, solid fraction and velocity) to 3D lattice.
        """
        # reset fields
        self.id.fill(-1)
        self.volfrac.fill(0.0)
        self.weight.fill(0.0)
        self.velsolid.fill(0.0)
        self.hydrotorque.fill(0.0)

        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # skip the boundary nodes
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            # loop through all grains
            for id in range(self.dem.gf.shape[0]):
                # convert grain position and size to lattice units
                xc = (self.dem.gf[id].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
                yc = (self.dem.gf[id].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
                zc = (self.dem.gf[id].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx
                r = (self.dem.gf[id].radius - 0.5 * self.unit.dx) / self.unit.dx

                # calculate the distance from cell center to grain center
                dist = ti.sqrt((xc - i) ** 2 + (yc - j) ** 2 + (zc - k) ** 2)

                # assign grain info to lattice if covered by grain
                if dist >= (r + 0.5 * ti.sqrt(3.0)):  # not covered by grain (3D diagonal)
                    continue
                elif dist <= (r - 0.5 * ti.sqrt(3.0)):  # fully covered by grain
                    self.id[i, j, k] = id
                    self.volfrac[i, j, k] = 1.0
                    self.weight[i, j, k] = 1.0  # weighting coefficient
                    self.compute_solid_vel(i, j, k, id, xc, yc, zc, dist)  # solid velocity
                    break
                else:  # partially covered by grain
                    # calculate the solid volume fraction using the 3D cell decomposition method
                    cnt = 0
                    for ii in range(5):
                        for jj in range(5):
                            for kk in range(5):
                                dist2 = ti.sqrt((xc - (i - 0.4 + 0.2 * ii)) ** 2 +
                                                (yc - (j - 0.4 + 0.2 * jj)) ** 2 +
                                                (zc - (k - 0.4 + 0.2 * kk)) ** 2)
                                if dist2 < r:
                                    cnt += 1

                    epsilon = cnt / 125.0  # 5^3 = 125 sub-cells in 3D

                    # count the maximum only
                    if epsilon > self.volfrac[i, j, k]:
                        self.id[i, j, k] = id
                        self.volfrac[i, j, k] = epsilon
                        self.compute_weight(i, j, k)  # calculate the weighting coefficient
                        self.compute_solid_vel(i, j, k, id, xc, yc, zc, dist)  # solid velocity

    # ==========================================#
    # ----- Calculate Weight Coefficient ----- #
    # ==========================================#

    @ti.func
    def compute_weight(self, i: int, j: int, k: int):
        """Calculates the weighting coefficient as a function of solid volume fraction and relaxation time.

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """
        coeff1 = self.volfrac[i, j, k] * (1. / self.omega[i, j, k] - 0.5)
        coeff2 = (1. - self.volfrac[i, j, k]) + (1. / self.omega[i, j, k] - 0.5)
        self.weight[i, j, k] = coeff1 / coeff2

    # ======================================#
    # ----- Calculate Solid Velocity ----- #
    # ======================================#
    @ti.func
    def compute_solid_vel(self, i: int, j: int, k: int, id: int, xc: float, yc: float, zc: float, dist: float):
        """Calculates the solid velocity at a 3D lattice cell.

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
            id (int): Grain ID.
            xc (float): Normalized x-coordinate of grain center.
            yc (float): Normalized y-coordinate of grain center.
            zc (float): Normalized z-coordinate of grain center.
            dist (float): Normalized distance between cell and grain centers.
        """
        # position vector from grain center to cell center
        r_vec = Vector3(i, j, k) - Vector3(xc, yc, zc)

        # compute velocity due to rotation: v = omega × r
        dist_phys = dist * self.unit.dx
        omega_cross_r = tm.cross(self.dem.gf[id].omega, r_vec * self.unit.dx)

        # total solid velocity = translational + rotational
        self.velsolid[i, j, k] = (self.dem.gf[id].velocity + omega_cross_r) * self.unit.dt / self.unit.dx

    # ===============================#
    # ----- Collision Process ----- #
    # ===============================#
    @ti.kernel
    def collide(self):
        """Weighted collision operator according to the solid volume fraction in 3D.

            fpc = f + B*Omega_s + (1-B)*Omega_f
            where Omega_s = f_inv - feq_inv + feq_solid - f
                  Omega_f = -omega*(f - feq)
        """
        # map grains to lattice
        self.grains2lattice()

        # reset hydrodynamic force and torque to zero
        self.hydroforce.fill(0.0)
        self.hydrotorque.fill(0.0)

        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # skip the boundary and gas nodes
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            # update the equilibrium state based on the fluid velocity
            self.compute_feq(i, j, k)

            # collision and record the hydrodynamic force
            if self.volfrac[i, j, k] > 0:
                self.collide_solid(i, j, k)  # collision for solid cells
            else:
                self.collide_fluid(i, j, k)  # collision for fluid cells

        # calculate hydrodynamic torques after force calculation
        self.compute_hydrotorque()

        # calculate hydrodynamic forces on grains
        self.lattice2grains()

    # ===============================================#
    # ----- Collision Process for Fluid Cells ----- #
    # ===============================================#
    @ti.func
    def collide_fluid(self, i: int, j: int, k: int):
        """BGK collision operator for fluid cells in 3D.

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """
        # collision (relax the distribution functions towards equilibrium)
        for q in ti.static(range(PSCLattice3D.Q)):
            self.fpc[i, j, k][q] = (1.0 - self.omega[i, j, k]) * self.f[i, j, k][q] + self.omega[i, j, k] * \
                                   self.feq[i, j, k][q]

    # ===============================================#
    # ----- Collision Process for Solid Cells ----- #
    # ===============================================#
    @ti.func
    def collide_solid(self, i: int, j: int, k: int):
        """BGK collision operator for solid cells in 3D.

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """
        # update the equilibrium state based on the solid velocity
        self.compute_feq_solid(i, j, k)

        # weighted collision
        for q in range(PSCLattice3D.Q):
            Omega_s = (self.f[i, j, k][PSCLattice3D.qinv[q]] - self.feq[i, j, k][PSCLattice3D.qinv[q]] +
                       self.feqsolid[i, j, k][q] - self.f[i, j, k][q])
            Omega_f = -self.omega[i, j, k] * (self.f[i, j, k][q] - self.feq[i, j, k][q])
            self.fpc[i, j, k][q] = self.f[i, j, k][q] + self.weight[i, j, k] * Omega_s + (
                        1.0 - self.weight[i, j, k]) * Omega_f

            # compute hydrodynamic force using the partially saturated cell method (in lattice units!)
            self.hydroforce[i, j, k] -= self.weight[i, j, k] * Omega_s * PSCLattice3D.c[q]

    # =========================================#
    # ----- Calculate Hydrodynamic Torque ----- #
    # =========================================#
    @ti.func
    def compute_hydrotorque(self):
        """Calculate hydrodynamic torque field according to the formula:
        T_j = sum_j [B_j * (x_j - x_g) × sum_i(Omega_i^s * c_i)]

        This should be called after collision step when hydroforce is computed.
        """
        # reset torque field
        self.hydrotorque.fill(0.0)

        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # skip boundary cells and cells without solid fraction
            if (self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP)) or (
                    self.volfrac[i, j, k] <= 0.0):
                continue

            # get grain information for this cell
            grain_id = self.id[i, j, k]
            if grain_id >= 0:
                # convert grain center position to lattice units
                xc = (self.dem.gf[grain_id].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
                yc = (self.dem.gf[grain_id].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
                zc = (self.dem.gf[grain_id].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx

                # position vector from grain center to lattice node: (x_j - x_g)
                r_vec = Vector3(i, j, k) - Vector3(xc, yc, zc)

                # the hydroforce[i,j,k] already contains: -B_j * sum_i(Omega_i^s * c_i)
                # so torque = B_j * (x_j - x_g) × sum_i(Omega_i^s * c_i) = -(x_j - x_g) × hydroforce[i,j,k]
                self.hydrotorque[i, j, k] = -tm.cross(r_vec, self.hydroforce[i, j, k])

    # ==============================================================#
    # ----- Calculate Local Equilibrium using Solid Velocity ----- #
    # ==============================================================#
    @ti.func
    def compute_feq_solid(self, i: int, j: int, k: int):
        """Calculate the equilibrium distribution functions based on the solid velocity in 3D.
            feq = w*rho*(1 + c.u/cs^2 + 0.5*(c.u)^2/(cs^4) - 0.5*u^2/cs^2)

            Note that the operation is purely local.

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """
        uv = tm.dot(self.velsolid[i, j, k], self.velsolid[i, j, k])
        for q in range(PSCLattice3D.Q):
            cu = tm.dot(PSCLattice3D.c[q], self.velsolid[i, j, k])
            self.feqsolid[i, j, k][q] = PSCLattice3D.w[q] * self.rho[i, j, k] * (
                        1.0 + 3.0 * cu + 4.5 * (cu ** 2) - 1.5 * uv)

    # ==========================================================#
    # ----- Calculate Hydrodynamic Force on DEM Particle ----- #
    # ==========================================================#

    @ti.func
    def lattice2grains(self):
        """Assigns hydrodynamic forces and torques on 3D DEM particles.
        """
        # reset force and torque to 0
        self.dem.gf.force_fluid.fill(0.0)
        self.dem.gf.moment_fluid.fill(0.0)

        for id in range(self.dem.gf.shape[0]):
            # convert grain position and size to lattice units
            xc = (self.dem.gf[id].position[0] - self.dem.config.domain.xmin + 0.5 * self.unit.dx) / self.unit.dx
            yc = (self.dem.gf[id].position[1] - self.dem.config.domain.ymin + 0.5 * self.unit.dx) / self.unit.dx
            zc = (self.dem.gf[id].position[2] - self.dem.config.domain.zmin + 0.5 * self.unit.dx) / self.unit.dx
            r = self.dem.gf[id].radius / self.unit.dx

            # extents of the lattice covered by the grain
            x_begin = max(int(xc - r), 0)
            x_end = min(int(xc + r + 2), self.Nx)
            y_begin = max(int(yc - r), 0)
            y_end = min(int(yc + r + 2), self.Ny)
            z_begin = max(int(zc - r), 0)
            z_end = min(int(zc + r + 2), self.Nz)

            for i in range(x_begin, x_end):
                for j in range(y_begin, y_end):
                    for k in range(z_begin, z_end):
                        # skip the boundary and gas nodes
                        if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                            continue

                        # accumulate the hydrodynamic force and torque if solid and the id matches
                        if (self.volfrac[i, j, k] > 0.0) and (self.id[i, j, k] == id):
                            # accumulate hydrodynamic force
                            Ff = self.hydroforce[i, j, k] * (self.unit.rho * (self.unit.dx ** 4)) / (
                                        self.unit.dt ** 2)
                            self.dem.gf[id].force_fluid += Ff

                            # accumulate hydrodynamic torque (convert from lattice units to physical units)
                            Tf = self.hydrotorque[i, j, k] * (self.unit.rho * (self.unit.dx ** 5)) / (
                                        self.unit.dt ** 2)
                            self.dem.gf[id].moment_fluid += Tf