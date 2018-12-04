"""

                Detector pixels mapping 1

This series of scripts calculate the pixel map between two detectors.
For each pixel in one detector, the scripts calculate the nearest
points and the corresponding weights of this pixel in the reciprocal
space with respect to the other pixel.

For this specific script, MPI is utilized. It aims to transform this pattern from
the cartesian grid to a r-theta grid

"""


import sys
import time

import h5py as h5
import numpy as np
from mpi4py import MPI

sys.path.append('/reg/neh/home/haoyuan/Documents/my_repos/Arsenal/')

import arsenal
import arsenal.geometry as ag
from arsenal import PsanaUtil

# Initialize the MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()
batch_num = comm_size

"""
Step one: All the node load the two detectors
"""
###################################################################################################
# Initialize amox34117 detector
# This is the desired detector. We want to map the other detector to this detector.
###################################################################################################
exp_line_desired = 'amo'
user_name_desired = 'haoyuan'

exp_name_desired = "amox34117"
run_num_desired = 193
det_name_desired = "pnccdFront"
photon_energy_desired = 1703.

# Get data source
(det_desired,
 run_desired,
 times_desired,
 evt_desired,
 info_dict_desired) = PsanaUtil.setup_exp(exp_name=exp_name_desired,
                                          run_num=run_num_desired,
                                          det_name=det_name_desired)
# Get pattern number
pattern_num_desired = len(times_desired)
print("There are {} patterns in this run in total.".format(pattern_num_desired))

###################################################################################################
# Initialize amo86615 detector
# We have data on this detector and we want to map this detector to the previous detector.
###################################################################################################

exp_line_exists = 'amo'
user_name_exists = 'haoyuan'

exp_name_exists = "amo86615"
run_num_exists = 197
det_name_exists = "pnccdBack"
photon_energy_exists = 1603.

# Get data source
(det_exists,
 run_exists,
 times_exists,
 evt_exists,
 info_dict_exists) = PsanaUtil.setup_exp(exp_name=exp_name_exists,
                                         run_num=run_num_exists,
                                         det_name=det_name_exists)
# Get pattern number
pattern_num_exists = len(times_exists)
print("There are {} patterns in this run in total.".format(pattern_num_exists))

###################################################################################################
# Detector Exists: Get detector pixel positions in reciprocal space
###################################################################################################
# Process the real space distance
pixel_position_exists = det_exists.coords_xyz(par=run_num_exists)
pixel_number_exists = int(np.prod(det_exists.pedestals(par=run_num_exists).shape))
pixel_position_exists_1d = np.zeros((pixel_number_exists, 3), dtype=np.float64)
for l in range(3):
    pixel_position_exists_1d[:, l] = np.reshape(pixel_position_exists[l], pixel_number_exists)

# Convert the meter
pixel_position_exists_1d *= 1e-6

# Get wave_vector
wavelength_exists = arsenal.physics.get_wavelength_m(photon_energy_ev=photon_energy_exists)
wavevector_exists = np.array([0, 0, np.pi * 2 / wavelength_exists])

# Get pixel position in reciprocal space
pixel_momentum_exists_1d = arsenal.physics.get_scattered_momentum(
    position_in_meter=pixel_position_exists_1d,
    wavevector_in_meter=wavevector_exists)
##################################################################################################
# Detector Desired: Get detector pixel positions in reciprocal space
###################################################################################################
# Process the real space distance
pixel_position_desired = det_desired.coords_xyz(par=run_num_desired)
pixel_number_desired = int(np.prod(det_desired.pedestals(par=run_num_desired).shape))
pixel_position_desired_1d = np.zeros((pixel_number_desired, 3), dtype=np.float64)
for l in range(3):
    pixel_position_desired_1d[:, l] = np.reshape(pixel_position_desired[l], pixel_number_desired)

# Convert to meter
pixel_position_desired_1d *= 1e-6

# Get wavevector
wavelength_desired = arsenal.physics.get_wavelength_m(photon_energy_ev=photon_energy_desired)
wavevector_desired = np.array([0, 0, 2 * np.pi / wavelength_desired])

# Get pixel position in reciprocal space
pixel_momentum_desired_1d = arsenal.physics.get_scattered_momentum(
    position_in_meter=pixel_position_desired_1d,
    wavevector_in_meter=wavevector_desired)
###################################################################################################
# Calculate the index map and weight map and save them to the output folder
###################################################################################################

"""
Step Two: Each node calculate its job
"""
# Generate job list
jobs_list = np.array_split(np.arange(pixel_number_exists), batch_num)

# This time, I would like to map the old detector detector data to the new detector. Therefore
# I need to calculate the corresponding index and weight for each pixels in the old detector to the
#  new detector
pixel_momentum_exists_1d_job = np.ascontiguousarray(pixel_momentum_exists_1d[jobs_list[comm_rank]])

tic = time.time()
print("Process {} begins to process the pixels.".format(comm_rank))
index_map_job, weight_map_job = ag.py_get_nearest_point_index_and_weight_3d(
    point_list_ref=pixel_momentum_desired_1d,
    point_list_new=pixel_momentum_exists_1d_job,
    nearest_neighbor_num=4)
toc = time.time()
print('Process {} takes {:.2f} seconds to finish {} pixels.'.format(comm_rank,
                                                                    toc - tic,
                                                                    jobs_list[comm_rank].shape[0]))

comm.Barrier()  # Synchronize
index_map_collect = comm.gather(index_map_job, root=0)
weight_map_collect = comm.gather(weight_map_job, root=0)

"""
Step Three: The master node collect and assemble the results
"""
if comm_rank == 0:
    index_map = np.concatenate(index_map_collect, axis=0)
    weight_map = np.concatenate(weight_map_collect, axis=0)

    with h5.File('../output/detector_map_{}.h5'.format(arsenal.util.time_stamp()), 'w') as h5file:
        h5file.create_dataset('index_map', data=index_map)
        h5file.create_dataset('weight_map', data=weight_map)

        desired_group = h5file.create_group('desired_detector')
        desired_group.create_dataset('exp line', data=exp_line_desired)
        desired_group.create_dataset('user name', data=user_name_desired)
        desired_group.create_dataset('run num', data=run_num_desired)
        desired_group.create_dataset('exp name', data=exp_name_desired)
        desired_group.create_dataset('detector name', data=det_name_desired)
        desired_group.create_dataset('photon energy', data=photon_energy_desired)

        desired_group = h5file.create_group('existing_detector')
        desired_group.create_dataset('exp line', data=exp_line_exists)
        desired_group.create_dataset('user name', data=user_name_exists)
        desired_group.create_dataset('run num', data=run_num_exists)
        desired_group.create_dataset('exp name', data=exp_name_exists)
        desired_group.create_dataset('detector name', data=det_name_exists)
        desired_group.create_dataset('photon energy', data=photon_energy_exists)
