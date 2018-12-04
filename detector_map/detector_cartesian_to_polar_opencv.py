"""

                Detector pixels mapping 1

This series of scripts calculate the pixel map between two detectors.
For each pixel in one detector, the scripts calculate the nearest
points and the corresponding weights of this pixel in the reciprocal
space with respect to the other pixel.

For this specific script, only a single cpu is used. It uses the opencv package.

"""

import sys

sys.path.append('/reg/neh/home/haoyuan/Documents/my_repos/Arsenal/')

import numpy as np
import arsenal.geometry as ag
from arsenal import PsanaUtil

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

##################################################################################################
# Initialize amo86615 detector 
# We have data on this detector and we want to map this detector to the previous detector.
#################################################################################################

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

##################################################################################################
# Get detector pixel positions
##################################################################################################
pixel_position_exists = det_exists.coords_xyz(par=run_num_exists)

pixel_number_exists = np.prod(det_exists.pedestals(par=run_num_exists).shape)

pixel_position_exists_1d = np.zeros((pixel_number_exists, 3), dtype=np.float64)
for l in range(3):
    pixel_position_exists_1d[:, l] = np.reshape(pixel_position_exists[l], pixel_number_exists)

pixel_position_desired = det_desired.coords_xyz(par=run_num_desired)

pixel_number_desired = np.prod(det_desired.pedestals(par=run_num_desired).shape)

pixel_position_desired_1d = np.zeros((pixel_number_desired, 3), dtype=np.float64)
for l in range(3):
    pixel_position_desired_1d[:, l] = np.reshape(pixel_position_desired[l], pixel_number_desired)

###################################################################################################
# Calculate the index map and weight map and save them to the output folder
###################################################################################################
"""
This time, I would like to map the old detector detector data to the new detector. Therefore
I need to calculate the corresponding index and weight for each pixels in the old detector 
to the new detector"""

(index_map,
 weight_map) = ag.py_get_nearest_point_index_and_weight_3d(point_list_ref=pixel_position_desired_1d,
                                                           point_list_new=pixel_position_exists_1d,
                                                           nearest_neighbor_num=4)

np.save('../output/index_map.npy', index_map)
np.save('../output/weight_map.npy', weight_map)
