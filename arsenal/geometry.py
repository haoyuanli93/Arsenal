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


########################################################################################################################
# Grid Geometry
########################################################################################################################
@jit(nonpython=True, parallel=True)
def get_nearest_point_and_weight(pixel_position_reciprocal, voxel_length):
    """
    In a 3D space, assume that we have a position vector (x,y,z) and we know the length each pixel represents,
    then calculate the nearest

    :param pixel_position_reciprocal: The position of each pixel in the reciprocal space in
    :param voxel_length:
    :return:
    """
    # convert_to_voxel_unit
    pixel_position_voxel_unit = pixel_position_reciprocal / voxel_length

    # Get the indexes of the eight nearest points.
    num_panel, num_x, num_y, _ = pixel_position_reciprocal.shape

    # Get one nearest neighbor
    _indexes = np.floor(pixel_position_voxel_unit).astype(np.int64)

    # Generate the holders
    indexes = np.zeros((num_panel, num_x, num_y, 8, 3), dtype=np.int64)
    weight = np.ones((num_panel, num_x, num_y, 8), dtype=np.float64)

    # Calculate the floors and the ceilings
    dfloor = pixel_position_voxel_unit - indexes
    dceiling = 1 - dfloor

    # Assign the correct values to the indexes
    indexes[:, :, :, 0, :] = _indexes

    indexes[:, :, :, 1, 0] = _indexes[:, :, :, 0]
    indexes[:, :, :, 1, 1] = _indexes[:, :, :, 1]
    indexes[:, :, :, 1, 2] = _indexes[:, :, :, 2] + 1

    indexes[:, :, :, 2, 0] = _indexes[:, :, :, 0]
    indexes[:, :, :, 2, 1] = _indexes[:, :, :, 1] + 1
    indexes[:, :, :, 2, 2] = _indexes[:, :, :, 2]

    indexes[:, :, :, 3, 0] = _indexes[:, :, :, 0]
    indexes[:, :, :, 3, 1] = _indexes[:, :, :, 1] + 1
    indexes[:, :, :, 3, 2] = _indexes[:, :, :, 2] + 1

    indexes[:, :, :, 4, 0] = _indexes[:, :, :, 0] + 1
    indexes[:, :, :, 4, 1] = _indexes[:, :, :, 1]
    indexes[:, :, :, 4, 2] = _indexes[:, :, :, 2]

    indexes[:, :, :, 5, 0] = _indexes[:, :, :, 0] + 1
    indexes[:, :, :, 5, 1] = _indexes[:, :, :, 1]
    indexes[:, :, :, 5, 2] = _indexes[:, :, :, 2] + 1

    indexes[:, :, :, 6, 0] = _indexes[:, :, :, 0] + 1
    indexes[:, :, :, 6, 1] = _indexes[:, :, :, 1] + 1
    indexes[:, :, :, 6, 2] = _indexes[:, :, :, 2]

    indexes[:, :, :, 7, :] = _indexes + 1

    # Assign the correct values to the weight
    weight[:, :, :, 0] = np.prod(dceiling, axis=-1)
    weight[:, :, :, 1] = dceiling[:, :, :, 0] * dceiling[:, :, :, 1] * dfloor[:, :, :, 2]
    weight[:, :, :, 2] = dceiling[:, :, :, 0] * dfloor[:, :, :, 1] * dceiling[:, :, :, 2]
    weight[:, :, :, 3] = dceiling[:, :, :, 0] * dfloor[:, :, :, 1] * dfloor[:, :, :, 2]
    weight[:, :, :, 4] = dfloor[:, :, :, 0] * dceiling[:, :, :, 1] * dceiling[:, :, :, 2]
    weight[:, :, :, 5] = dfloor[:, :, :, 0] * dceiling[:, :, :, 1] * dfloor[:, :, :, 2]
    weight[:, :, :, 6] = dfloor[:, :, :, 0] * dfloor[:, :, :, 1] * dceiling[:, :, :, 2]
    weight[:, :, :, 7] = np.prod(dfloor, axis=-1)

    return indexes, weight
