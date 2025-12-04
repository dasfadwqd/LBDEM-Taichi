"""
Parallel and serial prefix sum (exclusive scan) executor for Taichi.

Provides a fast parallel implementation assuming input length is a power of two,
as well as a fallback serial version. Used in broad-phase collision detection
to compute offsets for compact particle lists per spatial cell.
"""

import taichi as ti
from .utils import *


# ======================================
# Broad Phase Collision Detection
# ======================================
@ti.data_oriented
class PrefixSumExecutor:
    def __init__(self):
        self.tree: ti.SNode = None
        self.temp: ti.StructField = None

    @ti.kernel
    def serial(self, output: ti.template(), input: ti.template()):
        """Serial exclusive prefix sum """
        n = input.shape[0]
        output[0] = 0
        ti.loop_config(serialize=True)
        for i in range(1, n):
            output[i] = output[i - 1] + input[i - 1]

    @ti.kernel
    def _down(self, d: int, n: int, offset: ti.template(), output: ti.template()):
        """Down-sweep phase of parallel exclusive scan."""
        for i in range(n):
            if (i < d):
                ai = offset * (2 * i + 1) - 1
                bi = offset * (2 * i + 2) - 1
                output[bi] += output[ai]

    @ti.kernel
    def _up(self, d: int, n: int, offset: ti.template(), output: ti.template()):
        """Up-sweep phase of parallel exclusive scan."""
        for i in range(n):
            if (i < d):
                ai = offset * (2 * i + 1) - 1
                bi = offset * (2 * i + 2) - 1
                tmp = output[ai]
                output[ai] = output[bi]
                output[bi] += tmp

    @ti.kernel
    def _copy(self, n: int, output: ti.template(), input: ti.template()):
        """Copy input to output."""
        for i in range(n): output[i] = input[i]

    @ti.kernel
    def _copy_and_clear(self, n: int, npad: int, temp: ti.template(), input: ti.template()):
        """Copy input and zero-pad to npad."""
        for i in range(n): temp[i] = input[i]
        for i in range(n, npad): temp[i] = 0

    def parallel_fast(self, output, input, cal_total=False):
        """
        Parallel exclusive prefix sum for power-of-two-sized inputs.
        If cal_total=True, returns the total sum (last inclusive prefix).
        """
        ti.static_assert(next_pow2(input.shape[0]) == input.shape[0], "parallel_fast requires input count = 2**p")
        n: ti.i32 = input.shape[0]
        d = n >> 1
        self._copy(n, output, input)
        offset = 1
        while d > 0:
            self._down(d, n, offset, output)
            offset <<= 1
            d >>= 1

        output[n - 1] = 0
        d = 1
        while d < n:
            offset >>= 1
            self._up(d, n, offset, output)
            d <<= 1
        if cal_total:
            return output[n - 1] + input[n - 1]
