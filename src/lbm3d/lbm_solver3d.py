'''
A module that contains the basic lattice Boltzmann calculations for 3D D3Q19.
'''

import numpy as np
from typing import Tuple
from src.lbm3d.lbmutils import CellType, UnitConverter

import taichi as ti
import taichi.math as tm

# =====================================
# Type Definitions
# =====================================
Vector3 = ti.types.vector(3, float)

@ti.data_oriented
class BasicLattice3D:
    """LBM - D3Q19 - cssq = 1/3, w0 = 1/3, w1-w6 = 1/18, w7-w18 = 1/36.

    D3Q19 lattice structure:
    Direction 0: (0,0,0) - rest
    Directions 1-6: face neighbors (+/-x, +/-y, +/-z)
    Directions 7-18: edge neighbors (combinations of two axes)
    # D3Q19 lattice directions:
        # 0: (0,0,0), 1: (1,0,0), 2: (-1,0,0), 3: (0,1,0), 4: (0,-1,0), 5: (0,0,1), 6: (0,0,-1)
        # 7: (1,1,0), 8: (-1,-1,0), 9: (1,-1,0), 10: (-1,1,0)
        # 11: (1,0,1), 12: (-1,0,-1), 13: (1,0,-1), 14: (-1,0,1)
        # 15: (0,1,1), 16: (0,-1,-1), 17: (0,1,-1), 18: (0,-1,1)
     # Right-Hand Coordinate System (RHS)

           +Y
            |
            |
            |____ +X
           /
         /
      +Z

    """
    # D3Q19 lattice structure
    D = 3  # dimension
    Q = 19  # discrete directions

    # indices for opposite direction
    qinv = ti.field(int, shape=(Q,))
    qinv.from_numpy(np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=np.int32))

    # indices for symmetry operations (flip x, y, z)
    qsyx = ti.field(int, shape=(Q,))  # flip x
    qsyx.from_numpy(np.array([0, 2, 1, 3, 4, 5, 6, 10, 9, 8, 7, 14, 13, 12, 11, 15, 16, 17, 18], dtype=np.int32))

    qsyy = ti.field(int, shape=(Q,))  # flip y
    qsyy.from_numpy(np.array([0, 1, 2, 4, 3, 5, 6, 9, 10, 7, 8, 11, 12, 13, 14, 18, 17, 16, 15], dtype=np.int32))

    qsyz = ti.field(int, shape=(Q,))  # flip z
    qsyz.from_numpy(np.array([0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 13, 14, 11, 12, 17, 18, 15, 16], dtype=np.int32))

    # lattice velocities for D3Q19
    c = ti.Vector.field(n=D, dtype=float, shape=(19,))
    c.from_numpy(np.array([
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [-1, 0, 0],  # 2
        [0, 1, 0],  # 3
        [0, -1, 0],  # 4
        [0, 0, 1],  # 5
        [0, 0, -1],  # 6
        [1, 1, 0],  # 7
        [-1, -1, 0],  # 8
        [1, -1, 0],  # 9
        [-1, 1, 0],  # 10
        [1, 0, 1],  # 11
        [-1, 0, -1],  # 12
        [1, 0, -1],  # 13
        [-1, 0, 1],  # 14
        [0, 1, 1],  # 15
        [0, -1, -1],  # 16
        [0, 1, -1],  # 17
        [0, -1, 1]  # 18
    ], dtype=np.float64))

    # weights for D3Q19
    w = ti.field(float, shape=(Q,))
    w.from_numpy(np.array([
        1. / 3.,  # w0
        1. / 18., 1. / 18., 1. / 18., 1. / 18., 1. / 18., 1. / 18.,  # w1-w6 (face neighbors)
        1. / 36., 1. / 36., 1. / 36., 1. / 36.,  # w7-w10 (xy edges)
        1. / 36., 1. / 36., 1. / 36., 1. / 36.,  # w11-w14 (xz edges)
        1. / 36., 1. / 36., 1. / 36., 1. / 36.  # w15-w18 (yz edges)
    ], dtype=np.float64))

    # other parameters
    velmax = (2. / 3.) ** 0.5  # maximum velocity -> feq[0] > 0
    cssq = 1.0 / 3.0  # speed of sound squared

    # =========================#
    # ----- Constructor ----- #
    # =========================#
    def __init__(self, Nx: int, Ny: int, Nz: int, omega: float, dx: float, dt: float, rho: float):
        """Generate a 3D lattice with the mesoscopic and macroscopic variables.

        Args:
            Nx (int): Number of lattice nodes in x-direction.
            Ny (int): Number of lattice nodes in y-direction.
            Nz (int): Number of lattice nodes in z-direction.
            omega (float): Relaxation frequency.
            dx (float): Lattice spacing [m].
            dt (float): LBM time step [s].
            rho (float): Fluid densty [kg/m^3].
        """
        # parameters to define a lattice
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.omega = ti.field(float, shape=(Nx, Ny, Nz))
        self.omega.fill(omega)

        # density distribution functions
        self.f = ti.Vector.field(BasicLattice3D.Q, float, shape=(Nx, Ny, Nz))  # density distribution functions
        self.feq = ti.Vector.field(BasicLattice3D.Q, float,
                                   shape=(Nx, Ny, Nz))  # equilibrium density distribution functions
        self.fpc = ti.Vector.field(BasicLattice3D.Q, float,
                                   shape=(Nx, Ny, Nz))  # post-collision density distribution functions

        # macroscopic variables
        self.rho = ti.field(float, shape=(Nx, Ny, Nz))  # fluid density
        self.rho.fill(1.0)  # default to 1.0
        self.vel = ti.Vector.field(BasicLattice3D.D, float, shape=(Nx, Ny, Nz))  # flow velocity
        self.rate = ti.field(float, shape=(Nx, Ny, Nz))  # shear rate magnitude

        # cell types
        self.CT = ti.field(int, shape=(Nx, Ny, Nz))  # lattice cell
        self.CT.fill(CellType.FLUID)  # default to FLUID

        # unit converter
        self.unit = UnitConverter(dx, dt, rho)
        # stress tensor components (NEW)
        self.stress_xx = ti.field(float, shape=(Nx, Ny, Nz))  # normal stress in xx
        self.stress_yy = ti.field(float, shape=(Nx, Ny, Nz))  # normal stress in yy
        self.stress_zz = ti.field(float, shape=(Nx, Ny, Nz))  # normal stress in zz
        self.stress_xy = ti.field(float, shape=(Nx, Ny, Nz))  # shear stress in xy
        self.stress_xz = ti.field(float, shape=(Nx, Ny, Nz))  # shear stress in xz
        self.stress_yz = ti.field(float, shape=(Nx, Ny, Nz))  # shear stress in yz

        # derived stress quantities (NEW)
        self.shear_stress_mag = ti.field(float, shape=(Nx, Ny, Nz))  # magnitude of shear stress


    # ===============================================#
    # ----- Initialize Distribution Functions ----- #
    # ===============================================#
    @ti.kernel
    def initialize(self):
        """Initializes the density distributions with the equilibrium state.
        """
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # skip the boundary cells
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            # calculate the equilibrium distribution functions
            self.compute_feq(i, j, k)

            # assign the initial distribution function as the equilibrium state
            for q in ti.static(range(BasicLattice3D.Q)):
                self.f[i, j, k][q] = self.feq[i, j, k][q]

    # ===============================#
    # ----- Collision Process ----- #
    # ===============================#
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
                self.fpc[i, j, k][q] = (1.0 - self.omega[i, j, k]) * self.f[i, j, k][q] + self.omega[i, j, k] * \
                                       self.feq[i, j, k][q]

    # ===============================#
    # ----- Streaming Process ----- #
    # ===============================#
    @ti.kernel
    def stream(self):
        """Streaming process in LBM.

            Update the fluid density and velocity after the boundary conditions have been applied.
        """
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            # copy distribution function at rest
            self.f[i, j, k][0] = self.fpc[i, j, k][0]

            # propogate from the neighbours to the current cell
            for q in range(1, BasicLattice3D.Q):
                # index of the neighbouring cell
                iNext, jNext, kNext = self.compute_neigh_index(q, i, j, k)

                # streaming
                if self.CT[iNext, jNext, kNext] & (
                        CellType.FLUID | CellType.VEL_ZOUHE | CellType.VEL_EXIT |CellType.Pre_ZOUHE ):  # propogation
                    self.f[i, j, k][q] = self.fpc[iNext, jNext, kNext][q]
                elif self.CT[iNext, jNext, kNext] & CellType.OBSTACLE:  # bounce-back
                    self.f[i, j, k][q] = self.fpc[i, j, k][BasicLattice3D.qinv[q]]
                elif self.CT[iNext, jNext, kNext] & CellType.VEL_LADD:  # modified bounce-back
                    cu = tm.dot(BasicLattice3D.c[q], self.vel[iNext, jNext, kNext])
                    self.f[i, j, k][q] = self.fpc[i, j, k][BasicLattice3D.qinv[q]] + 2.0 * BasicLattice3D.w[q] * \
                                         self.rho[i, j, k] * cu / BasicLattice3D.cssq
                elif self.CT[iNext, jNext, kNext] & CellType.FREE_SLIP:  # specular reflection
                    if self.CT[iNext, jNext, kNext] & (CellType.LEFT | CellType.RIGHT):
                        self.f[i, j, k][q] = self.fpc[i, jNext, kNext][BasicLattice3D.qsyx[q]]
                    elif self.CT[iNext, jNext, kNext] & (CellType.BOTTOM | CellType.TOP):
                        self.f[i, j, k][q] = self.fpc[iNext, j, kNext][BasicLattice3D.qsyy[q]]
                    elif self.CT[iNext, jNext, kNext] & (CellType.BACK | CellType.FRONT):
                        self.f[i, j, k][q] = self.fpc[iNext, jNext, k][BasicLattice3D.qsyz[q]]

            # update density and velocity if fluid
            if self.CT[i, j, k] & CellType.FLUID: self.compute_rho_vel(i, j, k)

            # apply boundary conditions (wet node approaches)
            for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
                if self.CT[i, j, k] & CellType.VEL_ZOUHE:
                    self.vel_zouHe(i, j, k)
                elif self.CT[i, j, k] & CellType.VEL_EXIT:
                    self.bc_vel_exit(i, j, k)
                elif self.CT[i, j, k] & CellType.Pre_ZOUHE:
                    self.pre_zouHe(i, j, k)

    # =========================================#
    # ----- Calculate Local Equilibrium ----- #
    # =========================================#
    @ti.func
    def compute_feq(self, i: int, j: int, k: int):
        """Calculate the equilibrium distribution functions.
            feq = w*rho*(1 + c.u/cs^2 + 0.5*(c.u)^2/(cs^4) - 0.5*u^2/cs^2)

            Note that the operation is purely local.

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """
        uv = tm.dot(self.vel[i, j, k], self.vel[i, j, k])
        for q in ti.static(range(BasicLattice3D.Q)):
            cu = tm.dot(BasicLattice3D.c[q], self.vel[i, j, k])
            self.feq[i, j, k][q] = BasicLattice3D.w[q] * self.rho[i, j, k] * (
                        1.0 + 3.0 * cu + 4.5 * (cu ** 2) - 1.5 * uv)

    # ==============================================#
    # ----- Compute Fluid Density & Velocity ----- #
    # ==============================================#
    @ti.func
    def compute_rho_vel(self, i: int, j: int, k: int):
        """Update the density and velocity based on the conservation laws.

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """
        # set values to zero
        self.rho[i, j, k] = 0.0
        self.vel[i, j, k] = Vector3(0.0, 0.0, 0.0)

        # sum to get the macro-variables
        for q in range(BasicLattice3D.Q):
            self.rho[i, j, k] += self.f[i, j, k][q]  # density
            self.vel[i, j, k] += BasicLattice3D.c[q] * self.f[i, j, k][q]  # momentum

        # throw an error if the fluid density is less than zero
        assert self.rho[i, j, k] > 0, "Density of fluid cell [{}, {}, {}] must be positive!".format(i, j, k)

        # divide momentum by the density
        self.vel[i, j, k] /= self.rho[i, j, k]

        # throw an error if velocity is too large
        assert tm.dot(self.vel[i, j, k], self.vel[
            i, j, k]) ** 0.5 < BasicLattice3D.velmax, "Velocity of cell [{}, {}, {}] reaches the maximum!".format(i, j,
                                                                                                                  k)

    # ===============================================#
    # ----- Compute Indices of Neighbour Cell ----- #
    # ===============================================#
    @ti.func
    def compute_neigh_index(self, q: int, i: int, j: int, k: int) -> Tuple:
        """Return the indices of the neighbouring cells.

        Args:
            q (int): Direction index.
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.

        Returns:
            Tuple: Index of x, y, and z coordinate of the neighbouring cell.
        """
        # index of the neighbouring cell
        iNext = int(i - BasicLattice3D.c[q].x)
        jNext = int(j - BasicLattice3D.c[q].y)
        kNext = int(k - BasicLattice3D.c[q].z)

        # connect the tail to the head for the default periodic boundary condition
        if iNext >= self.Nx:
            iNext = 0
        elif iNext < 0:
            iNext = self.Nx - 1

        if jNext >= self.Ny:
            jNext = 0
        elif jNext < 0:
            jNext = self.Ny - 1

        if kNext >= self.Nz:
            kNext = 0
        elif kNext < 0:
            kNext = self.Nz - 1

        return iNext, jNext, kNext

    # ================================================#
    # ----- NEW: Compute Stress Tensor ----- #
    # ================================================#
    @ti.kernel
    def compute_stress(self):
        """Compute the stress tensor components using non-equilibrium distribution.

        The stress tensor is calculated as:
        σ_αβ = -(1 - ω*Δt/2) * Σ_i c_iα * c_iβ * (f_i - f_i^eq)

        For dimensionless LBM with Δt = 1:
        σ_αβ = -(1 - ω/2) * Σ_i c_iα * c_iβ * (f_i - f_i^eq)
        """
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # skip non-fluid cells
            if self.CT[i, j, k] & (CellType.OBSTACLE | CellType.VEL_LADD | CellType.FREE_SLIP):
                continue

            # relaxation factor: (1 - ω/2) for Δt = 1
            relax_factor = 1.0 - 0.5 * self.omega[i, j, k]

            # initialize stress components
            sxx = 0.0
            syy = 0.0
            szz = 0.0
            sxy = 0.0
            sxz = 0.0
            syz = 0.0

            # sum over all directions
            for q in range(BasicLattice3D.Q):
                # non-equilibrium distribution
                fneq = self.f[i, j, k][q] - self.feq[i, j, k][q]

                # lattice velocity components
                cx = BasicLattice3D.c[q].x
                cy = BasicLattice3D.c[q].y
                cz = BasicLattice3D.c[q].z

                # accumulate stress tensor components
                sxx += cx * cx * fneq
                syy += cy * cy * fneq
                szz += cz * cz * fneq
                sxy += cx * cy * fneq
                sxz += cx * cz * fneq
                syz += cy * cz * fneq

            # apply relaxation factor and negative sign
            self.stress_xx[i, j, k] = -relax_factor * sxx
            self.stress_yy[i, j, k] = -relax_factor * syy
            self.stress_zz[i, j, k] = -relax_factor * szz
            self.stress_xy[i, j, k] = -relax_factor * sxy
            self.stress_xz[i, j, k] = -relax_factor * sxz
            self.stress_yz[i, j, k] = -relax_factor * syz

            # compute shear stress magnitude (von Mises equivalent shear stress)
            # τ_eq = sqrt(0.5 * (τ_xy² + τ_xz² + τ_yz²))
            self.shear_stress_mag[i, j, k] = tm.sqrt(
                0.5 * (self.stress_xy[i, j, k] ** 2 +
                       self.stress_xz[i, j, k] ** 2 +
                       self.stress_yz[i, j, k] ** 2)
            )

    @ti.kernel
    def compute_plane_average_stress(self, axis: int, position: int) -> ti.types.vector(3, float):
        """计算特定平面上的平均剪切应力

        Args:
            axis: 0=x轴(yz平面), 1=y轴(xz平面), 2=z轴(xy平面)
            position: 沿该轴的位置索引

        Returns:
            ti.math.vec3: (avg_τ_xy, avg_τ_xz, avg_τ_yz) 该平面的平均剪切应力
        """
        sum_xy = 0.0
        sum_xz = 0.0
        sum_yz = 0.0
        count = 0

        if axis == 2:  # xy平面 (固定z)
            for i, j in ti.ndrange(self.Nx, self.Ny):
                if self.CT[i, j, position] & CellType.FLUID:
                    sum_xy += self.stress_xy[i, j, position]
                    sum_xz += self.stress_xz[i, j, position]
                    sum_yz += self.stress_yz[i, j, position]
                    count += 1

        elif axis == 1:  # xz平面 (固定y)
            for i, k in ti.ndrange(self.Nx, self.Nz):
                if self.CT[i, position, k] & CellType.FLUID:
                    sum_xy += self.stress_xy[i, position, k]
                    sum_xz += self.stress_xz[i, position, k]
                    sum_yz += self.stress_yz[i, position, k]
                    count += 1

        elif axis == 0:  # yz平面 (固定x)
            for j, k in ti.ndrange(self.Ny, self.Nz):
                if self.CT[position, j, k] & CellType.FLUID:
                    sum_xy += self.stress_xy[position, j, k]
                    sum_xz += self.stress_xz[position, j, k]
                    sum_yz += self.stress_yz[position, j, k]
                    count += 1

        # 计算平均值，避免除以0
        avg_xy = 0.0
        avg_xz = 0.0
        avg_yz = 0.0
        if count > 0:
            avg_xy = sum_xy / count
            avg_xz = sum_xz / count
            avg_yz = sum_yz / count

        return Vector3(avg_xy, avg_xz, avg_yz)
    # ===========================================#
    # ----- Compute Strain Rate Magnitude ----- #
    # ===========================================#
    @ti.func
    def compute_strain_rate(self, i: int, j: int, k: int):
        """Calculates the local strain-rate according to the momentum flux tensor.

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """
        # Calculate non-equilibrium momentum flux tensor components
        # Diagonal elements
        PiNeq_xx = 0.0
        PiNeq_yy = 0.0
        PiNeq_zz = 0.0

        # Off-diagonal elements
        PiNeq_xy = 0.0
        PiNeq_xz = 0.0
        PiNeq_yz = 0.0

        for q in range(BasicLattice3D.Q):
            fnoneq = self.f[i, j, k][q] - self.feq[i, j, k][q]
            cx = BasicLattice3D.c[q].x
            cy = BasicLattice3D.c[q].y
            cz = BasicLattice3D.c[q].z

            PiNeq_xx += fnoneq * cx * cx
            PiNeq_yy += fnoneq * cy * cy
            PiNeq_zz += fnoneq * cz * cz
            PiNeq_xy += fnoneq * cx * cy
            PiNeq_xz += fnoneq * cx * cz
            PiNeq_yz += fnoneq * cy * cz

        # calculate the magnitude of momentum flux tensor
        Pi = (2.0 * (PiNeq_xx ** 2 + PiNeq_yy ** 2 + PiNeq_zz ** 2 + 2.0 * (
                    PiNeq_xy ** 2 + PiNeq_xz ** 2 + PiNeq_yz ** 2))) ** 0.5

        # assign the local strain-rate
        self.rate[i, j, k] = self.omega[i, j, k] / (2.0 * BasicLattice3D.cssq * self.rho[i, j, k]) * Pi

    # ==================================#
    # ----- Zou & He Velocity BC ----- #
    # ==================================#
    @ti.func
    def bc_vel_zouhe(self, i: int, j: int, k: int):
        """Velocity boundary condition with the non-equilibrium bounce-back scheme.

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """
        if self.CT[i, j, k] & CellType.LEFT:
            # update density
            coeff = 1 / (1 - self.vel[i, j, k].x)
            self.rho[i, j, k] = coeff * (self.f[i, j, k][0] + self.f[i, j, k][3] + self.f[i, j, k][4] +
                                         self.f[i, j, k][5] + self.f[i, j, k][6] + self.f[i, j, k][15] +
                                         self.f[i, j, k][16] + self.f[i, j, k][17] + self.f[i, j, k][18] +
                                         2.0 * (self.f[i, j, k][2] + self.f[i, j, k][8] + self.f[i, j, k][10] +
                                                self.f[i, j, k][12] + self.f[i, j, k][14]))

            # update equilibrium state
            self.compute_feq(i, j, k)

            # calculate unknown populations
            self.f[i, j, k][1] = self.f[i, j, k][2] + (self.feq[i, j, k][1] - self.feq[i, j, k][2])
            self.f[i, j, k][7] = self.f[i, j, k][8] + (self.feq[i, j, k][7] - self.feq[i, j, k][8])
            self.f[i, j, k][9] = self.f[i, j, k][10] + (self.feq[i, j, k][9] - self.feq[i, j, k][10])
            self.f[i, j, k][11] = self.f[i, j, k][12] + (self.feq[i, j, k][11] - self.feq[i, j, k][12])
            self.f[i, j, k][13] = self.f[i, j, k][14] + (self.feq[i, j, k][13] - self.feq[i, j, k][14])

        elif self.CT[i, j, k] & CellType.RIGHT:
            coeff = 1 / (1 + self.vel[i, j, k].x)
            self.rho[i, j, k] = coeff * (self.f[i, j, k][0] + self.f[i, j, k][3] + self.f[i, j, k][4] +
                                         self.f[i, j, k][5] + self.f[i, j, k][6]  + self.f[i, j, k][15] +
                                         self.f[i, j, k][16] + self.f[i, j, k][17] + self.f[i, j, k][18] +
                                         2.0 * (self.f[i, j, k][1] + self.f[i, j, k][7] + self.f[i, j, k][9] +
                                                self.f[i, j, k][11] + self.f[i, j, k][13]))

            self.compute_feq(i, j, k)

            self.f[i, j, k][2] = self.f[i, j, k][1] + (self.feq[i, j, k][2] - self.feq[i, j, k][1])
            self.f[i, j, k][8] = self.f[i, j, k][7] + (self.feq[i, j, k][8] - self.feq[i, j, k][7])
            self.f[i, j, k][10] = self.f[i, j, k][9] + (self.feq[i, j, k][10] - self.feq[i, j, k][9])
            self.f[i, j, k][12] = self.f[i, j, k][11] + (self.feq[i, j, k][12] - self.feq[i, j, k][11])
            self.f[i, j, k][14] = self.f[i, j, k][13] + (self.feq[i, j, k][14] - self.feq[i, j, k][13])

        elif self.CT[i, j, k] & CellType.BOTTOM:
            coeff = 1 / (1 - self.vel[i, j, k].y)
            self.rho[i, j, k] = coeff * (self.f[i, j, k][0] + self.f[i, j, k][1] + self.f[i, j, k][2] +
                                         self.f[i, j, k][5] + self.f[i, j, k][6] + self.f[i, j, k][11] +
                                         self.f[i, j, k][12] + self.f[i, j, k][13] + self.f[i, j, k][14] +
                                         2.0 * (self.f[i, j, k][4] + self.f[i, j, k][8] + self.f[i, j, k][9] +
                                                self.f[i, j, k][16] + self.f[i, j, k][18]))

            self.compute_feq(i, j, k)

            self.f[i, j, k][3] = self.f[i, j, k][4] + (self.feq[i, j, k][3] - self.feq[i, j, k][4])
            self.f[i, j, k][7] = self.f[i, j, k][8] + (self.feq[i, j, k][7] - self.feq[i, j, k][8])
            self.f[i, j, k][10] = self.f[i, j, k][9] + (self.feq[i, j, k][10] - self.feq[i, j, k][9])
            self.f[i, j, k][15] = self.f[i, j, k][16] + (self.feq[i, j, k][15] - self.feq[i, j, k][16])
            self.f[i, j, k][17] = self.f[i, j, k][18] + (self.feq[i, j, k][17] - self.feq[i, j, k][18])

        elif self.CT[i, j, k] & CellType.TOP:
            coeff = 1 / (1 + self.vel[i, j, k].y)
            self.rho[i, j, k] = coeff * (self.f[i, j, k][0] + self.f[i, j, k][1] + self.f[i, j, k][2] +
                                         self.f[i, j, k][5] + self.f[i, j, k][6] + self.f[i, j, k][11] +
                                         self.f[i, j, k][12] + self.f[i, j, k][13] + self.f[i, j, k][14] +
                                         2.0 * (self.f[i, j, k][3] + self.f[i, j, k][7] + self.f[i, j, k][10] +
                                                self.f[i, j, k][15] + self.f[i, j, k][17]))

            self.compute_feq(i, j, k)

            self.f[i, j, k][4] = self.f[i, j, k][3] + (self.feq[i, j, k][4] - self.feq[i, j, k][3])
            self.f[i, j, k][8] = self.f[i, j, k][7] + (self.feq[i, j, k][8] - self.feq[i, j, k][7])
            self.f[i, j, k][9] = self.f[i, j, k][10] + (self.feq[i, j, k][9] - self.feq[i, j, k][10])
            self.f[i, j, k][16] = self.f[i, j, k][15] + (self.feq[i, j, k][16] - self.feq[i, j, k][15])
            self.f[i, j, k][18] = self.f[i, j, k][17] + (self.feq[i, j, k][18] - self.feq[i, j, k][17])

        elif self.CT[i, j, k] & CellType.BACK:
            coeff = 1 / (1 - self.vel[i, j, k].z)
            self.rho[i, j, k] = coeff * (self.f[i, j, k][0] + self.f[i, j, k][1] + self.f[i, j, k][2] +
                                         self.f[i, j, k][3] + self.f[i, j, k][4] + self.f[i, j, k][7] +
                                         self.f[i, j, k][8] + self.f[i, j, k][9] + self.f[i, j, k][10] +
                                         2.0 * (self.f[i, j, k][6] + self.f[i, j, k][12] + self.f[i, j, k][13] +
                                                self.f[i, j, k][16] + self.f[i, j, k][17]))

            self.compute_feq(i, j, k)

            self.f[i, j, k][5] = self.f[i, j, k][6] + (self.feq[i, j, k][5] - self.feq[i, j, k][6])
            self.f[i, j, k][11] = self.f[i, j, k][12] + (self.feq[i, j, k][11] - self.feq[i, j, k][12])
            self.f[i, j, k][14] = self.f[i, j, k][13] + (self.feq[i, j, k][14] - self.feq[i, j, k][13])
            self.f[i, j, k][15] = self.f[i, j, k][16] + (self.feq[i, j, k][15] - self.feq[i, j, k][16])
            self.f[i, j, k][18] = self.f[i, j, k][17] + (self.feq[i, j, k][18] - self.feq[i, j, k][17])

        elif self.CT[i, j, k] & CellType.FRONT:
            coeff = 1 / (1 + self.vel[i, j, k].z)
            self.rho[i, j, k] = coeff * (self.f[i, j, k][0] + self.f[i, j, k][1] + self.f[i, j, k][2] +
                                         self.f[i, j, k][3] + self.f[i, j, k][4] + self.f[i, j, k][7] +
                                         self.f[i, j, k][8] + self.f[i, j, k][9] + self.f[i, j, k][10] +
                                         2.0 * (self.f[i, j, k][5] + self.f[i, j, k][11] + self.f[i, j, k][14] +
                                                self.f[i, j, k][15] + self.f[i, j, k][18]))

            self.compute_feq(i, j, k)

            self.f[i, j, k][6] = self.f[i, j, k][5] + (self.feq[i, j, k][6] - self.feq[i, j, k][5])
            self.f[i, j, k][12] = self.f[i, j, k][11] + (self.feq[i, j, k][12] - self.feq[i, j, k][11])
            self.f[i, j, k][13] = self.f[i, j, k][14] + (self.feq[i, j, k][13] - self.feq[i, j, k][14])
            self.f[i, j, k][16] = self.f[i, j, k][15] + (self.feq[i, j, k][16] - self.feq[i, j, k][15])
            self.f[i, j, k][17] = self.f[i, j, k][18] + (self.feq[i, j, k][17] - self.feq[i, j, k][18])

    # =====================#
    # ----- Exit BC ----- #
    # =====================#
    @ti.func
    def bc_vel_exit(self, i: int, j: int, k: int):
        """Exit boundary condition with zero velocity gradient.

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """
        if self.CT[i, j, k] & CellType.LEFT:
            self.f[i, j, k][1] = self.f[i + 1, j, k][1]
            self.f[i, j, k][7] = self.f[i + 1, j, k][7]
            self.f[i, j, k][9] = self.f[i + 1, j, k][9]
            self.f[i, j, k][11] = self.f[i + 1, j, k][11]
            self.f[i, j, k][13] = self.f[i + 1, j, k][13]
        elif self.CT[i, j, k] & CellType.RIGHT:
            self.f[i, j, k][2] = self.f[i - 1, j, k][2]
            self.f[i, j, k][8] = self.f[i - 1, j, k][8]
            self.f[i, j, k][10] = self.f[i - 1, j, k][10]
            self.f[i, j, k][12] = self.f[i - 1, j, k][12]
            self.f[i, j, k][14] = self.f[i - 1, j, k][14]
        elif self.CT[i, j, k] & CellType.BOTTOM:
            self.f[i, j, k][3] = self.f[i, j + 1, k][3]
            self.f[i, j, k][7] = self.f[i, j + 1, k][7]
            self.f[i, j, k][10] = self.f[i, j + 1, k][10]
            self.f[i, j, k][15] = self.f[i, j + 1, k][15]
            self.f[i, j, k][17] = self.f[i, j + 1, k][17]
        elif self.CT[i, j, k] & CellType.TOP:
            self.f[i, j, k][4] = self.f[i, j - 1, k][4]
            self.f[i, j, k][8] = self.f[i, j - 1, k][8]
            self.f[i, j, k][9] = self.f[i, j - 1, k][9]
            self.f[i, j, k][16] = self.f[i, j - 1, k][16]
            self.f[i, j, k][18] = self.f[i, j - 1, k][18]
        elif self.CT[i, j, k] & CellType.BACK:
            self.f[i, j, k][5] = self.f[i, j, k + 1][5]
            self.f[i, j, k][11] = self.f[i, j, k + 1][11]
            self.f[i, j, k][14] = self.f[i, j, k + 1][14]
            self.f[i, j, k][15] = self.f[i, j, k + 1][15]
            self.f[i, j, k][18] = self.f[i, j, k + 1][18]
        elif self.CT[i, j, k] & CellType.FRONT:
            self.f[i, j, k][6] = self.f[i, j, k - 1][6]
            self.f[i, j, k][12] = self.f[i, j, k - 1][12]
            self.f[i, j, k][13] = self.f[i, j, k - 1][13]
            self.f[i, j, k][16] = self.f[i, j, k - 1][16]
            self.f[i, j, k][17] = self.f[i, j, k - 1][17]

        # update velocity and density
        self.compute_rho_vel(i, j, k)

    # ===========================================#
    # ----- Zou & He Velocity and Pressure ----- #
    # ===========================================#

    @ti.func
    def vel_zouHe(self, i: int, j: int, k: int):
        """Velocity boundary condition with the non-equilibrium bounce-back scheme.
        # Apply Zou-He scheme based on boundary position
        # D3Q19 lattice directions:
        # 0: (0,0,0), 1: (1,0,0), 2: (-1,0,0), 3: (0,1,0), 4: (0,-1,0), 5: (0,0,1), 6: (0,0,-1)
        # 7: (1,1,0), 8: (-1,-1,0), 9: (1,-1,0), 10: (-1,1,0)
        # 11: (1,0,1), 12: (-1,0,-1), 13: (1,0,-1), 14: (-1,0,1)
        # 15: (0,1,1), 16: (0,-1,-1), 17: (0,1,-1), 18: (0,-1,1)

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """
        ux = self.vel[i, j, k][0]
        uy = self.vel[i, j, k][1]
        uz = self.vel[i, j, k][2]
        if self.CT[i, j, k] & CellType.LEFT:
            # update density
            coeff = 1 / (1 - ux)
            self.rho[i, j, k] = coeff * (self.f[i, j, k][0] + self.f[i, j, k][3] + self.f[i, j, k][4] +
                                         self.f[i, j, k][5] + self.f[i, j, k][6] + self.f[i, j, k][15] +
                                         self.f[i, j, k][16] + self.f[i, j, k][17] + self.f[i, j, k][18] +
                                         2.0 * (self.f[i, j, k][2] + self.f[i, j, k][8] + self.f[i, j, k][10] +
                                                self.f[i, j, k][12] + self.f[i, j, k][14]))

            # Calculate unknown distributions using Zou-He equations
            Ny_x = 0.5 * (self.f[i, j, k][3] + self.f[i, j, k][15] + self.f[i, j, k][17] -
                          (self.f[i, j, k][4] + self.f[i, j, k][18] + self.f[i, j, k][16])) - 1 / 3 * self.rho[
                       i, j, k] * uy
            Nz_x = 0.5 * (self.f[i, j, k][5] + self.f[i, j, k][18] + self.f[i, j, k][15] -
                          (self.f[i, j, k][6] + self.f[i, j, k][17] + self.f[i, j, k][16])) - 1 / 3 * self.rho[
                       i, j, k] * uz

            # calculate unknown populations
            self.f[i, j, k][1] = self.f[i, j, k][2] + (1.0 / 3.0) * self.rho[i, j, k] * ux
            self.f[i, j, k][7] = self.f[i, j, k][8] + 1 / 6 * self.rho[i, j, k] * (ux + uy) - Ny_x
            self.f[i, j, k][9] = self.f[i, j, k][10] + 1 / 6 * self.rho[i, j, k] * (ux - uy) + Ny_x
            self.f[i, j, k][11] = self.f[i, j, k][12] + 1 / 6 * self.rho[i, j, k] * (ux + uz) - Nz_x
            self.f[i, j, k][13] = self.f[i, j, k][14] + 1 / 6 * self.rho[i, j, k] * (ux - uz) + Nz_x

        elif self.CT[i, j, k] & CellType.RIGHT:
            coeff = 1 / (1 + ux)
            self.rho[i, j, k] = coeff * (self.f[i, j, k][0] + self.f[i, j, k][3] + self.f[i, j, k][4] +
                                         self.f[i, j, k][5] + self.f[i, j, k][6] + self.f[i, j, k][15] +
                                         self.f[i, j, k][16] + self.f[i, j, k][17] + self.f[i, j, k][18] +
                                         2.0 * (self.f[i, j, k][1] + self.f[i, j, k][7] + self.f[i, j, k][9] +
                                                self.f[i, j, k][11] + self.f[i, j, k][13]))

            # Calculate unknown distributions using Zou-He equations
            Ny_x = 0.5 * (self.f[i, j, k][3] + self.f[i, j, k][15] + self.f[i, j, k][17] -
                          (self.f[i, j, k][4] + self.f[i, j, k][18] + self.f[i, j, k][16])) - 1 / 3 * self.rho[
                       i, j, k] * uy
            Nz_x = 0.5 * (self.f[i, j, k][5] + self.f[i, j, k][18] + self.f[i, j, k][15] -
                          (self.f[i, j, k][6] + self.f[i, j, k][17] + self.f[i, j, k][16])) - 1 / 3 * self.rho[
                       i, j, k] * uz

            self.f[i, j, k][2] = self.f[i, j, k][1] - (1.0 / 3.0) * self.rho[i, j, k] * ux
            self.f[i, j, k][8] = self.f[i, j, k][7] + 1 / 6 * self.rho[i, j, k] * (-ux - uy) + Ny_x
            self.f[i, j, k][10] = self.f[i, j, k][9] + 1 / 6 * self.rho[i, j, k] * (-ux + uy) - Ny_x
            self.f[i, j, k][12] = self.f[i, j, k][11] + 1 / 6 * self.rho[i, j, k] * (-ux - uz) + Nz_x
            self.f[i, j, k][14] = self.f[i, j, k][13] + 1 / 6 * self.rho[i, j, k] * (-ux + uz) - Nz_x

        elif self.CT[i, j, k] & CellType.BOTTOM:
            coeff = 1 / (1 - uy)
            self.rho[i, j, k] = coeff * (self.f[i, j, k][0] + self.f[i, j, k][1] + self.f[i, j, k][2] +
                                         self.f[i, j, k][5] + self.f[i, j, k][6] + self.f[i, j, k][11] +
                                         self.f[i, j, k][12] + self.f[i, j, k][13] + self.f[i, j, k][14] +
                                         2.0 * (self.f[i, j, k][4] + self.f[i, j, k][8] + self.f[i, j, k][9] +
                                                self.f[i, j, k][16] + self.f[i, j, k][18]))

            # Calculate unknown distributions using Zou-He equations
            Nx_y = 0.5 * (self.f[i, j, k][1] + self.f[i, j, k][11] + self.f[i, j, k][13] -
                          (self.f[i, j, k][2] + self.f[i, j, k][14] + self.f[i, j, k][12])) - 1 / 3 * self.rho[
                       i, j, k] * ux
            Nz_y = 0.5 * (self.f[i, j, k][5] + self.f[i, j, k][11] + self.f[i, j, k][14] -
                          (self.f[i, j, k][6] + self.f[i, j, k][13] + self.f[i, j, k][12])) - 1 / 3 * self.rho[
                       i, j, k] * uz

            self.f[i, j, k][3] = self.f[i, j, k][4] + (1.0 / 3.0) * self.rho[i, j, k] * uy
            self.f[i, j, k][7] = self.f[i, j, k][8] + 1 / 6 * self.rho[i, j, k] * (uy + ux) - Nx_y
            self.f[i, j, k][10] = self.f[i, j, k][9] + 1 / 6 * self.rho[i, j, k] * (uy - ux) + Nx_y
            self.f[i, j, k][15] = self.f[i, j, k][16] + 1 / 6 * self.rho[i, j, k] * (uy + uz) - Nz_y
            self.f[i, j, k][17] = self.f[i, j, k][18] + 1 / 6 * self.rho[i, j, k] * (uy - uz) + Nz_y

        elif self.CT[i, j, k] & CellType.TOP:
            coeff = 1 / (1 + uy)
            self.rho[i, j, k] = coeff * (self.f[i, j, k][0] + self.f[i, j, k][1] + self.f[i, j, k][2] +
                                         self.f[i, j, k][5] + self.f[i, j, k][6] + self.f[i, j, k][11] +
                                         self.f[i, j, k][12] + self.f[i, j, k][13] + self.f[i, j, k][14] +
                                         2.0 * (self.f[i, j, k][3] + self.f[i, j, k][7] + self.f[i, j, k][10] +
                                                self.f[i, j, k][15] + self.f[i, j, k][17]))

            # Calculate unknown distributions using Zou-He equations
            Nx_y = 0.5 * (self.f[i, j, k][1] + self.f[i, j, k][11] + self.f[i, j, k][13] -
                          (self.f[i, j, k][2] + self.f[i, j, k][14] + self.f[i, j, k][12])) - 1 / 3 * self.rho[
                       i, j, k] * ux
            Nz_y = 0.5 * (self.f[i, j, k][5] + self.f[i, j, k][11] + self.f[i, j, k][14] -
                          (self.f[i, j, k][6] + self.f[i, j, k][13] + self.f[i, j, k][12])) - 1 / 3 * self.rho[
                       i, j, k] * uz

            self.f[i, j, k][4] = self.f[i, j, k][3] - (1.0 / 3.0) * self.rho[i, j, k] * uy
            self.f[i, j, k][8] = self.f[i, j, k][7] + 1 / 6 * self.rho[i, j, k] * (-uy - ux) + Nx_y
            self.f[i, j, k][9] = self.f[i, j, k][10] + 1 / 6 * self.rho[i, j, k] * (-uy + ux) - Nx_y
            self.f[i, j, k][16] = self.f[i, j, k][15] + 1 / 6 * self.rho[i, j, k] * (-uy - uz) + Nz_y
            self.f[i, j, k][18] = self.f[i, j, k][17] + 1 / 6 * self.rho[i, j, k] * (-uy + uz) - Nz_y

        elif self.CT[i, j, k] & CellType.BACK:
            coeff = 1 / (1 - uz)
            self.rho[i, j, k] = coeff * (self.f[i, j, k][0] + self.f[i, j, k][1] + self.f[i, j, k][2] +
                                         self.f[i, j, k][3] + self.f[i, j, k][4] + self.f[i, j, k][7] +
                                         self.f[i, j, k][8] + self.f[i, j, k][9] + self.f[i, j, k][10] +
                                         2.0 * (self.f[i, j, k][6] + self.f[i, j, k][12] + self.f[i, j, k][13] +
                                                self.f[i, j, k][16] + self.f[i, j, k][17]))

            # Calculate unknown distributions using Zou-He equations
            Nx_z = 0.5 * (self.f[i, j, k][1] + self.f[i, j, k][7] + self.f[i, j, k][9] -
                          (self.f[i, j, k][2] + self.f[i, j, k][10] + self.f[i, j, k][8])) - 1 / 3 * self.rho[
                       i, j, k] * ux
            Ny_z = 0.5 * (self.f[i, j, k][3] + self.f[i, j, k][7] + self.f[i, j, k][10] -
                          (self.f[i, j, k][4] + self.f[i, j, k][9] + self.f[i, j, k][8])) - 1 / 3 * self.rho[
                       i, j, k] * uy

            self.f[i, j, k][5] = self.f[i, j, k][6] + (1.0 / 3.0) * self.rho[i, j, k] * uz
            self.f[i, j, k][11] = self.f[i, j, k][12] + 1 / 6 * self.rho[i, j, k] * (uz + ux) - Nx_z
            self.f[i, j, k][14] = self.f[i, j, k][13] + 1 / 6 * self.rho[i, j, k] * (uz - ux) + Nx_z
            self.f[i, j, k][15] = self.f[i, j, k][16] + 1 / 6 * self.rho[i, j, k] * (uz + uy) - Ny_z
            self.f[i, j, k][18] = self.f[i, j, k][17] + 1 / 6 * self.rho[i, j, k] * (uz - uy) + Ny_z

        elif self.CT[i, j, k] & CellType.FRONT:
            coeff = 1 / (1 + uz)
            self.rho[i, j, k] = coeff * (self.f[i, j, k][0] + self.f[i, j, k][1] + self.f[i, j, k][2] +
                                         self.f[i, j, k][3] + self.f[i, j, k][4] + self.f[i, j, k][7] +
                                         self.f[i, j, k][8] + self.f[i, j, k][9] + self.f[i, j, k][10] +
                                         2.0 * (self.f[i, j, k][5] + self.f[i, j, k][11] + self.f[i, j, k][14] +
                                                self.f[i, j, k][15] + self.f[i, j, k][18]))

            # Calculate unknown distributions using Zou-He equations
            Nx_z = 0.5 * (self.f[i, j, k][1] + self.f[i, j, k][7] + self.f[i, j, k][9] -
                          (self.f[i, j, k][2] + self.f[i, j, k][10] + self.f[i, j, k][8])) - 1 / 3 * self.rho[
                       i, j, k] * ux
            Ny_z = 0.5 * (self.f[i, j, k][3] + self.f[i, j, k][7] + self.f[i, j, k][10] -
                          (self.f[i, j, k][4] + self.f[i, j, k][9] + self.f[i, j, k][8])) - 1 / 3 * self.rho[
                       i, j, k] * uy

            self.f[i, j, k][6] = self.f[i, j, k][5] - (1.0 / 3.0) * self.rho[i, j, k] * uz
            self.f[i, j, k][12] = self.f[i, j, k][11] + 1 / 6 * self.rho[i, j, k] * (-uz - ux) + Nx_z
            self.f[i, j, k][13] = self.f[i, j, k][14] + 1 / 6 * self.rho[i, j, k] * (-uz + ux) - Nx_z
            self.f[i, j, k][16] = self.f[i, j, k][15] + 1 / 6 * self.rho[i, j, k] * (-uz - uy) + Ny_z
            self.f[i, j, k][17] = self.f[i, j, k][18] + 1 / 6 * self.rho[i, j, k] * (-uz + uy) - Ny_z

    @ti.func
    def pre_zouHe(self, i: int, j: int, k: int):
        """Pressure boundary condition with the non-equilibrium bounce-back scheme.
        # Apply Zou-He scheme based on boundary position
        # D3Q19 lattice directions:
        # 0: (0,0,0), 1: (1,0,0), 2: (-1,0,0), 3: (0,1,0), 4: (0,-1,0), 5: (0,0,1), 6: (0,0,-1)
        # 7: (1,1,0), 8: (-1,-1,0), 9: (1,-1,0), 10: (-1,1,0)
        # 11: (1,0,1), 12: (-1,0,-1), 13: (1,0,-1), 14: (-1,0,1)
        # 15: (0,1,1), 16: (0,-1,-1), 17: (0,1,-1), 18: (0,-1,1)

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """
        rho = self.rho[i, j, k]
        if self.CT[i, j, k] & CellType.LEFT:
            # update velocity
            self.vel[i, j, k][0] = 1.0 - (self.f[i, j, k][0] + self.f[i, j, k][3] + self.f[i, j, k][4] +
                                          self.f[i, j, k][5] + self.f[i, j, k][6] + self.f[i, j, k][15] +
                                          self.f[i, j, k][16] + self.f[i, j, k][17] + self.f[i, j, k][18] +
                                          2.0 * (self.f[i, j, k][2] + self.f[i, j, k][8] + self.f[i, j, k][10] +
                                                 self.f[i, j, k][12] + self.f[i, j, k][14])) / rho

            # Calculate unknown distributions using Zou-He equations
            Ny_x = 0.5 * (self.f[i, j, k][3] + self.f[i, j, k][15] + self.f[i, j, k][17] -
                          (self.f[i, j, k][4] + self.f[i, j, k][18] + self.f[i, j, k][16]))
            Nz_x = 0.5 * (self.f[i, j, k][5] + self.f[i, j, k][18] + self.f[i, j, k][15] -
                          (self.f[i, j, k][6] + self.f[i, j, k][17] + self.f[i, j, k][16]))

            # calculate unknown populations
            self.f[i, j, k][1] = self.f[i, j, k][2] + (1.0 / 3.0) * rho * self.vel[i, j, k][0]
            self.f[i, j, k][7] = self.f[i, j, k][8] + 1 / 6 * rho * self.vel[i, j, k][0] - Ny_x
            self.f[i, j, k][9] = self.f[i, j, k][10] + 1 / 6 * rho * self.vel[i, j, k][0] + Ny_x
            self.f[i, j, k][11] = self.f[i, j, k][12] + 1 / 6 * rho * self.vel[i, j, k][0] - Nz_x
            self.f[i, j, k][13] = self.f[i, j, k][14] + 1 / 6 * rho * self.vel[i, j, k][0] + Nz_x

        elif self.CT[i, j, k] & CellType.RIGHT:
            # update velocity
            self.vel[i, j, k][0] = (self.f[i, j, k][0] + self.f[i, j, k][3] + self.f[i, j, k][4] +
                                    self.f[i, j, k][5] + self.f[i, j, k][6] + self.f[i, j, k][15] +
                                    self.f[i, j, k][16] + self.f[i, j, k][17] + self.f[i, j, k][18] +
                                    2.0 * (self.f[i, j, k][1] + self.f[i, j, k][7] + self.f[i, j, k][9] +
                                           self.f[i, j, k][11] + self.f[i, j, k][13])) / rho - 1.0

            # Calculate unknown distributions using Zou-He equations
            Ny_x = 0.5 * (self.f[i, j, k][3] + self.f[i, j, k][15] + self.f[i, j, k][17] -
                          (self.f[i, j, k][4] + self.f[i, j, k][18] + self.f[i, j, k][16]))
            Nz_x = 0.5 * (self.f[i, j, k][5] + self.f[i, j, k][18] + self.f[i, j, k][15] -
                          (self.f[i, j, k][6] + self.f[i, j, k][17] + self.f[i, j, k][16]))

            self.f[i, j, k][2] = self.f[i, j, k][1] - (1.0 / 3.0) * rho * self.vel[i, j, k][0]
            self.f[i, j, k][8] = self.f[i, j, k][7] + 1 / 6 * rho * -self.vel[i, j, k][0] + Ny_x
            self.f[i, j, k][10] = self.f[i, j, k][9] + 1 / 6 * rho * -self.vel[i, j, k][0] - Ny_x
            self.f[i, j, k][12] = self.f[i, j, k][11] + 1 / 6 * rho * -self.vel[i, j, k][0] + Nz_x
            self.f[i, j, k][14] = self.f[i, j, k][13] + 1 / 6 * rho * -self.vel[i, j, k][0] - Nz_x

        elif self.CT[i, j, k] & CellType.BOTTOM:
            # update velocity
            self.vel[i, j, k][1] = 1.0 - (self.f[i, j, k][0] + self.f[i, j, k][1] + self.f[i, j, k][2] +
                                          self.f[i, j, k][5] + self.f[i, j, k][6] + self.f[i, j, k][11] +
                                          self.f[i, j, k][12] + self.f[i, j, k][13] + self.f[i, j, k][14] +
                                          2.0 * (self.f[i, j, k][4] + self.f[i, j, k][8] + self.f[i, j, k][9] +
                                                 self.f[i, j, k][16] + self.f[i, j, k][18])) / rho

            # Calculate unknown distributions using Zou-He equations
            Nx_y = 0.5 * (self.f[i, j, k][1] + self.f[i, j, k][11] + self.f[i, j, k][13] -
                          (self.f[i, j, k][2] + self.f[i, j, k][14] + self.f[i, j, k][12]))
            Nz_y = 0.5 * (self.f[i, j, k][5] + self.f[i, j, k][11] + self.f[i, j, k][14] -
                          (self.f[i, j, k][6] + self.f[i, j, k][13] + self.f[i, j, k][12]))

            self.f[i, j, k][3] = self.f[i, j, k][4] + (1.0 / 3.0) * rho * self.vel[i, j, k][1]
            self.f[i, j, k][7] = self.f[i, j, k][8] + 1 / 6 * rho * self.vel[i, j, k][1] - Nx_y
            self.f[i, j, k][10] = self.f[i, j, k][9] + 1 / 6 * rho * self.vel[i, j, k][1] + Nx_y
            self.f[i, j, k][15] = self.f[i, j, k][16] + 1 / 6 * rho * self.vel[i, j, k][1] - Nz_y
            self.f[i, j, k][17] = self.f[i, j, k][18] + 1 / 6 * rho * self.vel[i, j, k][1] + Nz_y

        elif self.CT[i, j, k] & CellType.TOP:
            # update velocity
            self.vel[i, j, k][1] = (self.f[i, j, k][0] + self.f[i, j, k][1] + self.f[i, j, k][2] +
                                    self.f[i, j, k][5] + self.f[i, j, k][6] + self.f[i, j, k][11] +
                                    self.f[i, j, k][12] + self.f[i, j, k][13] + self.f[i, j, k][14] +
                                    2.0 * (self.f[i, j, k][3] + self.f[i, j, k][7] + self.f[i, j, k][10] +
                                           self.f[i, j, k][15] + self.f[i, j, k][17])) / rho - 1.0

            # Calculate unknown distributions using Zou-He equations
            Nx_y = 0.5 * (self.f[i, j, k][1] + self.f[i, j, k][11] + self.f[i, j, k][13] -
                          (self.f[i, j, k][2] + self.f[i, j, k][14] + self.f[i, j, k][12]))
            Nz_y = 0.5 * (self.f[i, j, k][5] + self.f[i, j, k][11] + self.f[i, j, k][14] -
                          (self.f[i, j, k][6] + self.f[i, j, k][13] + self.f[i, j, k][12]))

            self.f[i, j, k][4] = self.f[i, j, k][3] - (1.0 / 3.0) * rho * self.vel[i, j, k][1]
            self.f[i, j, k][8] = self.f[i, j, k][7] + 1 / 6 * rho * -self.vel[i, j, k][1] + Nx_y
            self.f[i, j, k][9] = self.f[i, j, k][10] + 1 / 6 * rho * -self.vel[i, j, k][1] - Nx_y
            self.f[i, j, k][16] = self.f[i, j, k][15] + 1 / 6 * rho * -self.vel[i, j, k][1] + Nz_y
            self.f[i, j, k][18] = self.f[i, j, k][17] + 1 / 6 * rho * -self.vel[i, j, k][1] - Nz_y

        elif self.CT[i, j, k] & CellType.BACK:
            # update velocity
            self.vel[i, j, k][2] = 1.0 - (self.f[i, j, k][0] + self.f[i, j, k][1] + self.f[i, j, k][2] +
                                          self.f[i, j, k][3] + self.f[i, j, k][4] + self.f[i, j, k][7] +
                                          self.f[i, j, k][8] + self.f[i, j, k][9] + self.f[i, j, k][10] +
                                          2.0 * (self.f[i, j, k][6] + self.f[i, j, k][12] + self.f[i, j, k][13] +
                                                 self.f[i, j, k][16] + self.f[i, j, k][17])) / rho

            # Calculate unknown distributions using Zou-He equations
            Nx_z = 0.5 * (self.f[i, j, k][1] + self.f[i, j, k][7] + self.f[i, j, k][9] -
                          (self.f[i, j, k][2] + self.f[i, j, k][10] + self.f[i, j, k][8]))
            Ny_z = 0.5 * (self.f[i, j, k][3] + self.f[i, j, k][7] + self.f[i, j, k][10] -
                          (self.f[i, j, k][4] + self.f[i, j, k][9] + self.f[i, j, k][8]))

            self.f[i, j, k][5] = self.f[i, j, k][6] + (1.0 / 3.0) * rho * self.vel[i, j, k][2]
            self.f[i, j, k][11] = self.f[i, j, k][12] + 1 / 6 * rho * self.vel[i, j, k][2] - Nx_z
            self.f[i, j, k][14] = self.f[i, j, k][13] + 1 / 6 * rho * self.vel[i, j, k][2] + Nx_z
            self.f[i, j, k][15] = self.f[i, j, k][16] + 1 / 6 * rho * self.vel[i, j, k][2] - Ny_z
            self.f[i, j, k][18] = self.f[i, j, k][17] + 1 / 6 * rho * self.vel[i, j, k][2] + Ny_z

        elif self.CT[i, j, k] & CellType.FRONT:
            # update velocity
            self.vel[i, j, k][2] = (self.f[i, j, k][0] + self.f[i, j, k][1] + self.f[i, j, k][2] +
                                    self.f[i, j, k][3] + self.f[i, j, k][4] + self.f[i, j, k][7] +
                                    self.f[i, j, k][8] + self.f[i, j, k][9] + self.f[i, j, k][10] +
                                    2.0 * (self.f[i, j, k][5] + self.f[i, j, k][11] + self.f[i, j, k][14] +
                                           self.f[i, j, k][15] + self.f[i, j, k][18])) / rho - 1.0

            # Calculate unknown distributions using Zou-He equations
            Nx_z = 0.5 * (self.f[i, j, k][1] + self.f[i, j, k][7] + self.f[i, j, k][9] -
                          (self.f[i, j, k][2] + self.f[i, j, k][10] + self.f[i, j, k][8]))
            Ny_z = 0.5 * (self.f[i, j, k][3] + self.f[i, j, k][7] + self.f[i, j, k][10] -
                          (self.f[i, j, k][4] + self.f[i, j, k][9] + self.f[i, j, k][8]))

            self.f[i, j, k][6] = self.f[i, j, k][5] - (1.0 / 3.0) * rho * self.vel[i, j, k][2]
            self.f[i, j, k][12] = self.f[i, j, k][11] + 1 / 6 * rho * -self.vel[i, j, k][2] + Nx_z
            self.f[i, j, k][13] = self.f[i, j, k][14] + 1 / 6 * rho * -self.vel[i, j, k][2] - Nx_z
            self.f[i, j, k][16] = self.f[i, j, k][15] + 1 / 6 * rho * -self.vel[i, j, k][2] + Ny_z
            self.f[i, j, k][17] = self.f[i, j, k][18] + 1 / 6 * rho * -self.vel[i, j, k][2] - Ny_z
'''
    # =============================#
    # ----- Partial-slip BC ----- #
    # =============================#
    @ti.func
    def bc_partial_slip(self, i: int, j: int, k: int):
        """Partial slip boundary condition for 3D.

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """
        # Implementation depends on specific partial slip model
        # This is a placeholder for future implementation
        pass

    # ============================#
    # ----- Navier Slip BC ----- #
    # ============================#
    @ti.func
    def bc_navier_slip(self, i: int, j: int, k: int):
        """Navier slip boundary condition for 3D.

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.
        """
        # Implementation depends on specific Navier slip model
        # This is a placeholder for future implementation
        pass


    # =======================================#
    # ----- Additional 3D Utility Methods ----- #
    # =======================================#
    @ti.func
    def get_velocity_magnitude(self, i: int, j: int, k: int) -> float:
        """Get velocity magnitude at a given cell.

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.

        Returns:
            float: Velocity magnitude
        """
        return tm.sqrt(tm.dot(self.vel[i, j, k], self.vel[i, j, k]))

    @ti.func
    def get_vorticity(self, i: int, j: int, k: int) -> ti.Vector:
        """Calculate vorticity at a given cell using finite differences.

        Args:
            i (int): Index of x-coordinate.
            j (int): Index of y-coordinate.
            k (int): Index of z-coordinate.

        Returns:
            ti.Vector: Vorticity vector (omega_x, omega_y, omega_z)
        """
        # Get neighboring indices with periodic boundary conditions
        ip = (i + 1) % self.Nx
        im = (i - 1 + self.Nx) % self.Nx
        jp = (j + 1) % self.Ny
        jm = (j - 1 + self.Ny) % self.Ny
        kp = (k + 1) % self.Nz
        km = (k - 1 + self.Nz) % self.Nz

        # Calculate vorticity components using central differences
        omega_x = 0.5 * (self.vel[i, jp, k].z - self.vel[i, jm, k].z) - 0.5 * (
                    self.vel[i, j, kp].y - self.vel[i, j, km].y)
        omega_y = 0.5 * (self.vel[i, j, kp].x - self.vel[i, j, km].x) - 0.5 * (
                    self.vel[ip, j, k].z - self.vel[im, j, k].z)
        omega_z = 0.5 * (self.vel[ip, j, k].y - self.vel[im, j, k].y) - 0.5 * (
                    self.vel[i, jp, k].x - self.vel[i, jm, k].x)

        return ti.Vector([omega_x, omega_y, omega_z])

    @ti.kernel
    def compute_all_strain_rates(self):
        """Compute strain rates for all fluid cells."""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & CellType.FLUID:
                self.compute_strain_rate(i, j, k)

    @ti.kernel
    def get_kinetic_energy(self) -> float:
        """Calculate total kinetic energy of the fluid domain.

        Returns:
            float: Total kinetic energy
        """
        total_ke = 0.0
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & CellType.FLUID:
                vel_sq = tm.dot(self.vel[i, j, k], self.vel[i, j, k])
                total_ke += 0.5 * self.rho[i, j, k] * vel_sq
        return total_ke

    @ti.kernel
    def get_total_mass(self) -> float:
        """Calculate total mass in the fluid domain.

        Returns:
            float: Total mass
        """
        total_mass = 0.0
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.CT[i, j, k] & CellType.FLUID:
                total_mass += self.rho[i, j, k]
        return total_mass
        
'''