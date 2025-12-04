import taichi as ti
import numpy as np
# =====================================
# Utils
# =====================================
def np2csv(name, data):
    np.savetxt(name + ".csv", data, delimiter=",")

def cal_neighbor_search_radius(max_radius):
    return max_radius * 1.1

def next_pow2(x):
    x -= 1
    x |= (x >> 1)
    x |= (x >> 2)
    x |= (x >> 4)
    x |= (x >> 8)
    x |= (x >> 16)
    return x + 1

def round32(n: ti.i32):
    if (n % 32 == 0):
        return n
    else:
        return ((n >> 5) + 1) << 5

@ti.func
def round32d(n: ti.i32):
    if (n % 32 == 0):
        return n
    else:
        return ((n >> 5) + 1) << 5