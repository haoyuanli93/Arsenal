from numba import jit
import os, sys
import numpy as np
import mpi4py.MPI as MPI
import time

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


@jit
def make_rot_quat(quaternion):
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


@jit
def slice_merge(quaternion, _slice, model3d, weight, detector, size):
    num_pix = _slice.shape[0]
    mask = detector[:, 4]
    center = size / 2.

    rot = np.zeros([3, 3])

    rot = make_rot_quat(quaternion)

    for t in range(0, num_pix):

        if (mask[t] > 1.1):
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


@jit
def post_process(model3d, weight, size):
    for k in range(size ** 3):
        if (weight[k] > 0):
            model3d[k] /= weight[k]
    return model3d


if __name__ == "__main__":

    # comm = MPI.COMM_WORLD  
    # comm_rank = comm.Get_rank()  
    # comm_size = comm.Get_size()  

    # get orientation array from file
    orientation = None
    forientation = '/reg/d/psdm/amo/amo86615/scratch/zhensu/recon_0008/data/orientations/orientations_010.bin'
    with open(forientation, 'r') as f:
        orientation = np.fromfile(f, '=i4')  # 1D array

    # get detector array from file
    fdetector = '/reg/d/psdm/amo/amo86615/scratch/zhensu/recon_0008/data/det_sim.dat'
    f = open(fdetector, "r")
    det_content = f.readlines()
    f.close()

    num_pix = int(det_content[0])
    detector = np.zeros([num_pix, 5])
    for idx, val in enumerate(det_content[1:]):
        valsplit = val.split(' ', 4)
        detector[idx, 0] = float(valsplit[0])
        detector[idx, 1] = float(valsplit[1])
        detector[idx, 2] = float(valsplit[2])
        detector[idx, 3] = float(valsplit[3])
        detector[idx, 4] = int(valsplit[4])

    # get photon array
    fphotons = '/reg/d/psdm/amo/amo86615/scratch/zhensu/recon_0008/data/photons.emc'
    ones = []
    multi = []
    place_ones = []
    place_multi = []
    count_multi = []
    with open(fphotons, 'r') as f:
        num_data = int(np.fromfile(f, '=i4', 1))
        num_pix = int(np.fromfile(f, '=i4', 1))
        np.fromfile(f, '=i1', 1016)
        for k in range(num_data):
            ones.append(np.fromfile(f, '=i4', 1))
        for k in range(num_data):
            multi.append(np.fromfile(f, '=i4', 1))
        for k in range(num_data):
            place_ones.append(np.fromfile(f, '=i4', ones[k]))
        for k in range(num_data):
            place_multi.append(np.fromfile(f, '=i4', multi[k]))
        for k in range(num_data):
            count_multi.append(np.fromfile(f, '=i4', multi[k]))

    # get scale array
    fscale = '/reg/d/psdm/amo/amo86615/scratch/zhensu/recon_0008/data/scale/scale_010.dat'
    f = open(fscale, 'r')
    scale_content = f.readlines()
    f.close()
    scale = []
    for each in scale_content:
        scale.append(float(each))
    scale = np.array(scale)  # 1D numpy array

    # get quaternion array
    fquaternion = '/reg/d/psdm/amo/amo86615/res/zhensu/recon_0001/data/quat.dat'
    f = open(fquaternion, 'r')
    quat_content = f.readlines()
    f.close()
    num_quat = int(quat_content[0])
    quaternion = np.zeros([num_quat, 4])
    # quat_weight = np.zeros(num_quat)

    for idx, val in enumerate(quat_content[1:]):
        valsplit = val.split(' ', 4)
        quaternion[idx, :] = [valsplit[0], valsplit[1], valsplit[2], valsplit[3]]
        # quat_weight[idx] = valsplit[4]

    local_offset = np.linspace(0, num_data, comm_size + 1).astype('int')

    num_samp = 211
    model3d = np.zeros(num_samp ** 3).astype('d')
    weight = np.zeros(num_samp ** 3).astype('d')

    print('rank ' + str(comm_rank) + ' is ready')

    if (comm_rank == 0):
        tic = time.time()
    for idx in range(local_offset[comm_rank], local_offset[comm_rank + 1]):
        if (comm_rank == 0) and (idx % 5 == 0):
            print(idx)
        _slice = np.zeros(num_pix)
        _slice[place_ones[idx]] = 1.
        _slice[place_multi[idx]] = count_multi[idx] * 1.0
        _slice = _slice / scale[idx]
        quat = quaternion[orientation[idx]]
        [model3d, weight] = slice_merge(quat, _slice, model3d, weight, detector, num_samp)

    if (comm_rank == 0):
        toc = time.time()
        print
        "time (s): ", toc - tic

    total_intensity = None
    total_weight = None

    if (comm_rank == 0):
        total_intensity = np.empty([comm_size, num_samp ** 3], dtype='d')
        total_weight = np.empty([comm_size, num_samp ** 3], dtype='d')

    comm.Gather(model3d, total_intensity, root=0)
    comm.Gather(weight, total_weight, root=0)

    if (comm_rank == 0):
        toc = time.time()
        print
        "time (s): ", toc - tic

    if comm_rank == 0:
        total_intensity = total_intensity.sum(axis=0)
        total_weight = total_weight.sum(axis=0)
        total_intensity = post_process(total_intensity, total_weight, num_samp)

        toc = time.time()
        print
        "time (s): ", toc - tic

        f = open('/reg/neh/home5/zhensu/Test/intensity5.dat', 'w')
        for i in range(num_samp ** 3):
            f.write(str(total_intensity[i]) + '\n')
        f.close()
        print
        "Intensity data has been written to file"
        toc = time.time()
        print
        "time (s): ", toc - tic 





