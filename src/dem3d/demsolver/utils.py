import taichi as ti
#=====================================
# Type Definition
#=====================================
Vector2 = ti.types.vector(2, int)
Vector3 = ti.types.vector(3, float)
Vector4 = ti.types.vector(4, float)
Vector3i = ti.types.vector(3, int)
Vector2i = ti.types.vector(2, int)
Matrix3x3 = ti.types.matrix(3, 3, float)
DEMMatrix = Matrix3x3
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


@ti.func
def Zero3x3() -> Matrix3x3:
    return Matrix3x3([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# Add a math function: quaternion to rotation matrix
# References:
# https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
# Lines 511-534, https://github.com/CFDEMproject/LIGGGHTS-PUBLIC/blob/master/src/math_extra_liggghts_nonspherical.h
@ti.func
def quat2RotMatrix(quat: Vector4) -> Matrix3x3:
    # w i j k
    # 0 1 2 3
    w2 = quat[0] * quat[0]
    i2 = quat[1] * quat[1]
    j2 = quat[2] * quat[2]
    k2 = quat[3] * quat[3]

    twoij = 2.0 * quat[1] * quat[2]
    twoik = 2.0 * quat[1] * quat[3]
    twojk = 2.0 * quat[2] * quat[3]
    twoiw = 2.0 * quat[1] * quat[0]
    twojw = 2.0 * quat[2] * quat[0]
    twokw = 2.0 * quat[3] * quat[0]

    result = Zero3x3()
    result[0, 0] = w2 + i2 - j2 - k2
    result[0, 1] = twoij - twokw
    result[0, 2] = twojw + twoik
    result[1, 0] = twoij + twokw
    result[1, 1] = w2 - i2 + j2 - k2
    result[1, 2] = twojk - twoiw
    result[2, 0] = twoik - twojw
    result[2, 1] = twojk + twoiw
    result[2, 2] = w2 - i2 - j2 + k2

    return result