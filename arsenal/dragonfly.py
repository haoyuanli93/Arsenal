import numpy as np
from numba import jit

from arsenal import geometry as geom


def get_detector_from_file(fdetector):
    """
    Extract the detector array from the file.

    :param fdetector: The file of the detector in dragonfly
    :return: The detector numpy array
    """
    # get detector array from file
    with open(fdetector, "r") as dfile:
        det_content = dfile.readlines()

    # Construct the detector array
    num_pix = int(det_content[0])
    detector = np.zeros([num_pix, 5])
    for idx, val in enumerate(det_content[1:]):
        valsplit = val.split(' ', 4)
        detector[idx, 0] = float(valsplit[0])
        detector[idx, 1] = float(valsplit[1])
        detector[idx, 2] = float(valsplit[2])
        detector[idx, 3] = float(valsplit[3])
        detector[idx, 4] = int(valsplit[4])

    return detector


def get_photons_from_emc(fphotons, pixel_num):
    """
    Parse the emc file to reconstruct the photons array.

    :param fphotons: The emc file containing the photons array.
    :param pixel_num: The pixel number.
    :return:The photons array.
    """
    # Holder for photon array
    ones = []
    multi = []
    place_ones = []
    place_multi = []
    count_multi = []

    # Extract the photons info
    with open(fphotons, 'r') as f:
        pattern_num = int(np.fromfile(f, '=i4', 1))
        for k in range(pattern_num):
            ones.append(np.fromfile(f, '=i4', 1))
            multi.append(np.fromfile(f, '=i4', 1))
            place_ones.append(np.fromfile(f, '=i4', ones[k]))
            place_multi.append(np.fromfile(f, '=i4', multi[k]))
            count_multi.append(np.fromfile(f, '=i4', multi[k]))

    # Reconstruct the photons array
    photons = np.zeros((pattern_num, pixel_num), dtype=np.float64)

    for idx in range(pattern_num):
        # Use a holder to store the data
        holder = np.zeros(pixel_num)
        holder[place_ones[idx]] = 1.
        holder[place_multi[idx]] = count_multi[idx]

        photons[idx] = holder[:]

    return photons


def get_quaternion(fquaternion):
    """
    Read the quaternion from the file.

    :param fquaternion: The file containing the quaternion
    :return:
    """
    # get quaternion array
    with open(fquaternion, 'r') as f:
        quat_content = f.readlines()

    num_quat = int(quat_content[0])
    quaternion = np.zeros([num_quat, 4])

    for idx, val in enumerate(quat_content[1:]):
        valsplit = val.split(' ', 4)
        quaternion[idx, :] = [valsplit[0], valsplit[1], valsplit[2], valsplit[3]]


def get_scale(fscale):
    """
    Read the scaling factor from the file.

    :param fscale: The file containing the scaling factor.
    :return:
    """
    with open(fscale, 'r') as f:
        scale_content = f.readlines()

    # Cast to a numpy array
    scale = []
    for each in scale_content:
        scale.append(float(each))

    scale = np.array(scale)
    return scale


@jit
def merge_single_slice(quaternion, _slice, model3d, weight, detector, size):
    """
    Merge the slice to the volume.

    :param quaternion: The quaternion array
    :param _slice: The slice to save

    :param detector:
    :param size:
    :return:
    """
    num_pix = _slice.shape[0]
    mask = detector[:, 4]
    center = size / 2.

    rot = geom.quaternion_to_rotation_matrix(quaternion)

    for t in range(0, num_pix):

        if mask[t] > 1.1:
            continue

        rot_pix = np.zeros(3)

        for i in range(0, 3):
            for j in range(0, 3):
                rot_pix[i] += rot[i, j] * detector[t, j]
            rot_pix[i] += center

        tx = rot_pix[0]
        ty = rot_pix[1]
        tz = rot_pix[2]

        x = int(tx)
        y = int(ty)
        z = int(tz)

        if (x < 0) or x > (size - 2) or (y < 0) or y > (size - 2) or (z < 0) or z > (size - 2):
            continue

        fx = tx - x
        fy = ty - y
        fz = tz - z
        cx = 1. - fx
        cy = 1. - fy
        cz = 1. - fz

        # Correct for solid angle and polarization
        _slice[t] /= detector[t, 3]
        w = _slice[t]

        f = cx * cy * cz
        weight[x * size * size + y * size + z] += f
        model3d[x * size * size + y * size + z] += f * w

        f = cx * cy * fz
        weight[x * size * size + y * size + ((z + 1) % size)] += f
        model3d[x * size * size + y * size + ((z + 1) % size)] += f * w

        f = cx * fy * cz
        weight[x * size * size + ((y + 1) % size) * size + z] += f
        model3d[x * size * size + ((y + 1) % size) * size + z] += f * w

        f = cx * fy * fz
        weight[x * size * size + ((y + 1) % size) * size + ((z + 1) % size)] += f
        model3d[x * size * size + ((y + 1) % size) * size + ((z + 1) % size)] += f * w

        f = fx * cy * cz
        weight[((x + 1) % size) * size * size + y * size + z] += f
        model3d[((x + 1) % size) * size * size + y * size + z] += f * w

        f = fx * cy * fz
        weight[((x + 1) % size) * size * size + y * size + ((z + 1) % size)] += f
        model3d[((x + 1) % size) * size * size + y * size + ((z + 1) % size)] += f * w

        f = fx * fy * cz
        weight[((x + 1) % size) * size * size + ((y + 1) % size) * size + z] += f
        model3d[((x + 1) % size) * size * size + ((y + 1) % size) * size + z] += f * w

        f = fx * fy * fz
        weight[((x + 1) % size) * size * size + ((y + 1) % size) * size + ((z + 1) % size)] += f
        model3d[((x + 1) % size) * size * size + ((y + 1) % size) * size + ((z + 1) % size)] += f * w

    return np.array([model3d, weight])
