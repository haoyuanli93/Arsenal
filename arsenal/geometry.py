import numpy as np
from numba import jit

from arsenal.rare import cummulate_product_with_local_exclusion_dim1


########################################################################################################################
# General geometric functions
########################################################################################################################
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
# General geometric functions
########################################################################################################################
def get_solid_angle(pixel_area_m, pixel_position_m, pixel_orientation, reference_point_m):
    """
    Calculate the solid angle for each pixel with respect to the reference point.

    I assume the following situation:

                                           |    --- pixel 1
                                            \   --- pixel 2
                    ----> ref point.
                                            /   --- pixel 3
                                           _    --- pixel 4

    Each pixel has an independent orientation and area and position. This position is the absolute position
    with respect to some known origin. The reference is the interaction center. The solid angle is calculated
    with respect to this reference point.

    :param pixel_area_m: This is the pixel area in meter. This is a 1d array.
    :param pixel_position_m: This the pixel position in meter. This is a 2D array. [pixel number, 3]
                              [[x0,y0,z0] -- pixel 0
                               [x1,y1,z1] -- pixel 1
                               ...
                               ]
    :param pixel_orientation:  This is the normal direction of each pixel with respect the origin.
                                This is a 2D array. [pixel number, 3]

                              [[x0,y0,z0] -- pixel 0
                               [x1,y1,z1] -- pixel 1
                               ...
                               ]
    :param reference_point_m:  This is the position of the reference point with respect to the origin in meter.
    :return: The solid angle of each pixel.
    """

    # Calculate the position of the pixels with respect to the reference point
    relative_position = pixel_position_m - reference_point_m[np.newaxis, :]

    # Calculate the distance and direction of each pixel in real space w.r.t. the reference point
    pixel_distances = np.sqrt(np.sum(np.square(relative_position), axis=-1))
    pixel_directions = np.divide(relative_position, pixel_distances[:, np.newaxis])

    # Calculate the cosine of the angle between the normal direction of each pixel
    # and the direction between the pixel and the reference point. The underlying formula is
    #
    #     solid_angle = area*cos(theta) / distance^2 = area * <normal, direction>  / distance^2
    #
    # Here, area*cos(theta) is the area perpendicular to the radius.

    true_area = np.multiply(pixel_area_m, np.sum(np.multiply(pixel_orientation, pixel_directions), axis=-1))
    # Because the normal direction can be either positive or negative, but
    # solid angle can only be positive, take absolute values.
    true_area = np.abs(true_area)

    solid_angle = np.divide(true_area, np.square(pixel_distances))
    return solid_angle


########################################################################################################################
# General pixel interpolation
########################################################################################################################
@jit('void(int64, float64[:], float64[:,:], float64[:])', nopython=True, parallel=True)
def get_distance_list(pixel_num, single_point, reference_point_list, output):
    """
    Calculate the distances between the single points and
    the reference point list and store them in the output variable.

    :param pixel_num: The number of pixels to calculate.
    :param single_point:  The single point to investigate
    :param reference_point_list: The reference point
    :param output: The output variable to store the result
    :return: None
    """
    # Calculate the difference
    for l in range(pixel_num):
        output[l] = np.sqrt((reference_point_list[l, 0] - single_point[0]) ** 2 +
                            (reference_point_list[l, 1] - single_point[1]) ** 2 +
                            (reference_point_list[l, 2] - single_point[2]) ** 2)


@jit('void(int64, int64, int64, float64[:,:], float64[:,:], int64[:,:], float64[:,:])', nopython=True, parallel=True)
def get_nearest_point_and_distance_arbitrary_mesh_3d(point_num_new,
                                                     point_num_ref,
                                                     nearest_neighbor_num,
                                                     old_point_list,
                                                     new_point_list,
                                                     nn_index_holder,
                                                     nn_distance_holder):
    """
    Calculate the nearest neighbor of points in the new point list with respect to the old point list.

    :param point_num_new:
    :param point_num_ref:
    :param nearest_neighbor_num:
    :param old_point_list:
    :param new_point_list::
    :param nn_distance_holder:
    :param nn_index_holder:
    :return:
    """

    # It turns out that this can be extremely slow considering that we have 1024*1024 pixels
    distance_holder = np.ascontiguousarray(np.ones(point_num_ref, dtype=np.float64))

    for l in range(point_num_new):
        # Step 1: Calculate the distance matrix.
        # Calculate the distance
        get_distance_list(pixel_num=point_num_ref,
                          single_point=new_point_list[l, :],
                          reference_point_list=old_point_list,
                          output=distance_holder)

        # Step 2: Sort the distance and find the nearest neighbor index
        nn_index_holder[l, :] = np.argsort(distance_holder)[:nearest_neighbor_num]

        # Step 3: Extract the distance
        for m in range(nearest_neighbor_num):
            nn_distance_holder[l, m] = distance_holder[m]


def py_get_nearest_point_index_and_weight_3d(point_list_ref, point_list_new, nearest_neighbor_num):
    """
    Get the nearest point index and weight in old_point_list for each point in the new_point_list

    Assume that the contribution is proportional to the product of the other distances

    :param point_list_ref: [point_number_old, 3]  dtype=float32
    :param point_list_new: [point_number_new, 3]  dtype=float32
    :param nearest_neighbor_num:
    :return: index, weight
    """

    # Create variables
    point_num_new = point_list_new.shape[0]
    point_num_ref = point_list_ref.shape[0]

    nn_index_holder = np.ascontiguousarray(np.ones((point_num_new, nearest_neighbor_num), dtype=np.int64))
    nn_distance_holder = np.ascontiguousarray(np.ones((point_num_new, nearest_neighbor_num), dtype=np.float64))

    # Calculate the nearest neighbor
    get_nearest_point_and_distance_arbitrary_mesh_3d(point_num_new=point_num_new,
                                                     point_num_ref=point_num_ref,
                                                     nearest_neighbor_num=nearest_neighbor_num,
                                                     old_point_list=point_list_ref,
                                                     new_point_list=point_list_new,
                                                     nn_index_holder=nn_index_holder,
                                                     nn_distance_holder=nn_distance_holder)

    # Calculate the weight
    nn_weight_holder = cummulate_product_with_local_exclusion_dim1(num_dim1=nearest_neighbor_num,
                                                                   arry=nn_distance_holder)
    tmp = np.sum(nn_weight_holder, axis=-1)  # To normalize the weight to get a probability distribution

    # Normalize the weight to get the probability distribution
    np.divide(nn_weight_holder, tmp[:, np.newaxis], out=nn_weight_holder)
    return nn_index_holder, nn_weight_holder


@jit(nopython=True, parallel=True)
def detector_mapping(pixel_num, nearest_neighbor_num, index_map, weight, raw_pattern, new_pattern):
    """
    perform the detector geometry map with the index map the weight info and the raw pattern to map

    Notice that here, one does not scale the result with the weight
    summation of the weight on each pixel in the new pattern

    :param pixel_num:
    :param nearest_neighbor_num:
    :param index_map:
    :param weight:
    :param raw_pattern:
    :param new_pattern: Output
    :return:
    """

    for l in range(pixel_num):
        for n in range(nearest_neighbor_num):
            new_pattern[index_map[l, n]] += raw_pattern[l] * weight[l, n]


########################################################################################################################
# Geometry function for special cases
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
