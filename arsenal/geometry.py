import numpy as np
from numba import jit


@jit
def quaternion_to_rotation_matrix(quaternion):
    """
    Generate a rotation matrix from the quaternion value.
    :param quaternion:
    :return:
    """
    rot = np.zeros([3, 3])

    q0 = quaternion[0]
    q1 = quaternion[1]
    q2 = quaternion[2]
    q3 = quaternion[3]

    q01 = q0 * q1
    q02 = q0 * q2
    q03 = q0 * q3
    q11 = q1 * q1
    q12 = q1 * q2
    q13 = q1 * q3
    q22 = q2 * q2
    q23 = q2 * q3
    q33 = q3 * q3

    rot[0, 0] = (1. - 2. * (q22 + q33))
    rot[0, 1] = 2. * (q12 + q03)
    rot[0, 2] = 2. * (q13 - q02)
    rot[1, 0] = 2. * (q12 - q03)
    rot[1, 1] = (1. - 2. * (q11 + q33))
    rot[1, 2] = 2. * (q01 + q23)
    rot[2, 0] = 2. * (q02 + q13)
    rot[2, 1] = 2. * (q23 - q01)
    rot[2, 2] = (1. - 2. * (q11 + q22))

    return rot
