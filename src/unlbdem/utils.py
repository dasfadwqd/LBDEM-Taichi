'''
A module that contains the calculation function for the unresolved LBM-DEM .
'''
import taichi as ti

# =================================#
# ----- Calculation function ----- #
# =================================#

class Interpolation:
    def __init__(self):
        '''
        Three-point delta function for smooth particle-to-grid interpolation.

        This kernel function provides smooth weight distribution with:
        - Compact support: w(r) = 0 for r > 1.5
        - Continuity: C0 continuous (value continuous, derivative may jump)
        - Normalization: Σw = 1.0 when integrated over all space
        - Symmetry: w(r) = w(-r)

        The kernel consists of two piecewise regions:

        Region 1 (0 ≤ r < 0.5): Near field, high weight
            w(r) = [1 + (-3r² + 1)/2] / 3
                 = (1 - 1.5r²) / 3 + 1/3

        Region 2 (0.5 ≤ r ≤ 1.5): Far field, decaying weight
            w(r) = [5 - 3r - √(-3(1-r)² + 1)] / 6

        **Key Properties:**
        - w(0) = 2/3 ≈ 0.667 (maximum weight at particle center)
        - w(0.5) = 7/24 ≈ 0.292 (weight at half lattice spacing)
        - w(1.5) = 0 (zero weight beyond 1.5 lattice spacings)
        - Continuous at r = 0.5

        **Physical Interpretation:**
        The kernel mimics a "smeared out" particle distribution, allowing each
        particle to influence nearby cells smoothly. The 1.5 lattice support
        means each particle typically affects ~(3³) = 27 cells in 3D.

        **Normalization Property:**
        For a particle in free space (no boundaries), the sum of weights over
        all influenced cells equals 1.0:
            Σ w(||x_cell - x_particle||) = 1.0

        This ensures conservation of mass/volume when mapping particles to grid.

        Args:
            r (float): Distance from particle center in lattice units

        Returns:
            float: Interpolation weight w(r) ∈ [0, 2/3]

        '''

    @ti.func
    def threedelta(self, r) -> float:
        a = 0.0
        if r < 0.5:
            x = -3.0 * r ** 2 + 1.0
            a = (1.0 + x * 0.5) / 3.0
        elif r <= 1.5:
            x = -3.0 * (1.0 - r) ** 2 + 1.0
            a = (5.0 - 3.0 * r - ti.sqrt(x)) / 6.0
        return a



