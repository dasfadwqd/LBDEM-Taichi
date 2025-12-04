"""
Broad-Phase Collision Detection (BPCD) using spatial hashing with 3D Morton (Z-order) encoding.

This module implements an efficient grid-based broad-phase collision detection system 
for 3D discrete element simulations.
Particles are assigned to spatial cells of size
4 × max_radius. To maintain spatial locality in the hash table, 3D grid coordinates 
are mapped to 1D hash indices using a 30-bit Morton code (10 bits per axis), followed 
by modulo table size. 

The pipeline consists of three stages:
1. Count particles per hash cell (atomic increment).
2. Compute cell offsets via parallel prefix sum.
3. Emit particle indices into a compact list and perform neighbor search.
"""

import taichi as ti
from .prefixsum import PrefixSumExecutor
from .utils import *

#=====================================
# Type Definitions
#=====================================

Vector3 = ti.types.vector(3, float)
Vector3i = ti.types.vector(3, int)


@ti.data_oriented
class BPCD:
    """
    Broad-Phase Collision Detection using Morton-coded spatial hashing.
    """

    @ti.dataclass
    class HashCell:
        offset: int    # Start index in the compact particle_id list (set by prefix sum)
        count: int     # Number of particles hashed to this cell
        current: int   # Atomic counter used during particle insertion

    def __init__(self, particle_count: int, hash_table_size: int, max_radius: float, domain_min: Vector3):
        self.cell_size = max_radius * 4.0
        self.domain_min = domain_min
        self.hash_table = BPCD.HashCell.field(shape=hash_table_size)
        self.particle_id = ti.field(dtype=int, shape=particle_count)
        self.pse = PrefixSumExecutor()

    @staticmethod
    def create(particle_count: int, max_radius: float, domain_min: Vector3, domain_max: Vector3):
        """
        Factory method to initialize BPCD with a hash table size based on domain volume.
        Table size is rounded up to the next power of two for efficient modulo via bit masking.
        """
        v = (domain_max - domain_min) / (4.0 * max_radius)
        size = int(v[0] * v[1] * v[2])
        size = next_pow2(size)
        return BPCD(particle_count, size, max_radius, domain_min)

    def detect_collision(self, positions, collision_resolve_callback=None):
        """
        Main entry point for broad-phase collision detection.

        Parameters:
            positions: Taichi field of Vector3 (not Vector2 as in outdated comment)
            collision_resolve_callback: Callable(i, j) invoked for each unique candidate pair (i < j)
        """
        self._setup_collision(positions)
        self.pse.parallel_fast(self.hash_table.offset, self.hash_table.count)
        self._put_particles(positions)
        self._solve_collision(positions, collision_resolve_callback)

    @ti.func
    def _count_particles(self, position: Vector3):
        """Atomically increment particle count in the hash cell for the given position."""
        ht = ti.static(self.hash_table)
        ti.atomic_add(ht[self.hash_codef(position)].count, 1)

    @ti.kernel
    def _put_particles(self, positions: ti.template()):
        """Insert particle indices into the compact particle_id list per hash cell."""
        ht = ti.static(self.hash_table)
        pid = ti.static(self.particle_id)
        for i in positions:
            hash_cell = self.hash_codef(positions[i])
            loc = ti.atomic_add(ht[hash_cell].current, 1)
            offset = ht[hash_cell].offset
            pid[offset + loc] = i

    @ti.func
    def _clear_hash_cell(self, i: int):
        """Reset hash cell counters for a new detection cycle."""
        ht = ti.static(self.hash_table)
        ht[i].offset = 0
        ht[i].current = 0
        ht[i].count = 0

    @ti.kernel
    def _setup_collision(self, positions: ti.template()):
        """Clear all hash cells and count particles per cell."""
        ht = ti.static(self.hash_table)
        for i in ht:
            self._clear_hash_cell(i)
        for i in positions:
            self._count_particles(positions[i])

    @ti.kernel
    def _solve_collision(self, positions: ti.template(), collision_resolve_callback: ti.template()):
        """
        Search for collision candidates in up to 8 neighboring cells.
        The neighbor selection is adaptive: only the octant aligned with the particle's
        offset from its cell center is probed, reducing the number of cells from 27 to 8.
        """
        ht = ti.static(self.hash_table)
        for i in positions:
            o = positions[i]
            ijk = self.cell(o)
            center = self.cell_center(ijk)
            dxyz = Vector3i(0, 0, 0)

            # Determine which octant the particle lies in relative to cell center
            for k in ti.static(range(3)):
                dxyz[k] = 1 if (o[k] > center[k]) else -1

            # Generate the 8 potentially interacting neighbor cells
            cells = [
                ijk,
                ijk + Vector3i(dxyz[0], 0, 0),
                ijk + Vector3i(0, dxyz[1], 0),
                ijk + Vector3i(0, 0, dxyz[2]),
                ijk + Vector3i(0, dxyz[1], dxyz[2]),
                ijk + Vector3i(dxyz[0], 0, dxyz[2]),
                ijk + Vector3i(dxyz[0], dxyz[1], 0),
                ijk + dxyz
            ]

            for k in ti.static(range(len(cells))):
                hash_cell = ht[self.hash_code(cells[k])]
                # if(hash_cell.offset + hash_cell.count > self.particle_id.shape[0]): print(f"i = {i}, cell={self.hash_code(cells[k])}, offset = {hash_cell.offset}, count={hash_cell.count}")
                if hash_cell.count > 0:
                    for idx in range(hash_cell.offset, hash_cell.offset + hash_cell.count):
                        pid = self.particle_id[idx]

                        if pid > i:
                            collision_resolve_callback(i, pid)

    @ti.func
    def morton3d32(x: int, y: int, z: int) -> int:
        """
        Compute 30-bit 3D Morton code (Z-order curve) by interleaving bits of x, y, z.
        Based on the bit-manipulation method from:
        https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number
        """
        x &= 0x3ff
        x = (x | x << 16) & 0x30000ff
        x = (x | x << 8) & 0x300f00f
        x = (x | x << 4) & 0x30c30c3
        x = (x | x << 2) & 0x9249249

        y &= 0x3ff
        y = (y | y << 16) & 0x30000ff
        y = (y | y << 8) & 0x300f00f
        y = (y | y << 4) & 0x30c30c3
        y = (y | y << 2) & 0x9249249

        z &= 0x3ff
        z = (z | z << 16) & 0x30000ff
        z = (z | z << 8) & 0x300f00f
        z = (z | z << 4) & 0x30c30c3
        z = (z | z << 2) & 0x9249249

        return x | (y << 1) | (z << 2)

    @ti.func
    def hash_codef(self, xyz: Vector3) -> int:
        """Compute hash code from a world-space position."""
        return self.hash_code(self.cell(xyz))

    @ti.func
    def hash_code(self, ijk: Vector3i) -> int:
        """Map integer grid coordinates to a hash table index using Morton code and modulo."""
        return BPCD.morton3d32(ijk[0], ijk[1], ijk[2]) % self.hash_table.shape[0]

    @ti.func
    def cell(self, xyz: Vector3) -> Vector3i:
        """Map a world-space position to integer grid coordinates."""
        return ti.floor((xyz - self.domain_min) / self.cell_size, dtype=int)

    @ti.func
    def cell_center(self, ijk: Vector3i) -> Vector3:
        """Compute the world-space center of a grid cell given its integer coordinates."""
        ret = Vector3(0, 0, 0)
        for i in ti.static(range(3)):
            ret[i] = (ijk[i] + 0.5) * self.cell_size + self.domain_min[i]
        return ret