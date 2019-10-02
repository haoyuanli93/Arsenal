import sys
import numpy as np
import h5py as h5
import time

sys.path.append('/reg/neh/home5/haoyuan/Documents/my_repos/Arsenal')
from arsenal import PsanaUtil

####################################################################################################
# [USER] Load the run numbers and indexes
####################################################################################################
source = np.load("../output/single_hit_run_index_amo87215.npy")
run_num_single = source[:, 0]
index_single = source[:, 1]

# Get total pattern number
pattern_num = run_num_single.shape[0]

# Divide the whole dataset into several smaller data
batch_number = int(pattern_num / 128)

run_lists = np.array_split(run_num_single, batch_number)
index_lists = np.array_split(index_single, batch_number)

# The experiment info
exp_line = 'amo'
exp_name = 'amo87215'
user_name = 'haoyuan'
det_name = 'pnccdFront'

# Add a tag
tag = 'single_hits'

# Construct the output address
output_address = '../output/'

####################################################################################################
# [USER] Get an example pattern
####################################################################################################
# Get data source
det, run, times, evt, info_dict = PsanaUtil.setup_exp(exp_name=exp_name,
                                                      run_num=run_num_single[0],
                                                      det_name=det_name)
shape_stack = info_dict['Pattern stack shape']
shape_2d = info_dict['2D pattern shape']

####################################################################################################
# [USER] Loop through all the sub list
####################################################################################################

for bidx in range(batch_number):  # batch index
    run_list = run_lists[bidx]
    index_list = index_lists[bidx]

    # Get image number in the sub list
    pattern_num = run_list.shape[0]

    # Create a holder for the 2d pattern
    adu_2d = np.zeros((pattern_num,) + shape_2d, dtype=np.float64)
    adu_stack = np.zeros((pattern_num,) + shape_stack, dtype=np.float64)

    photon_2d = np.zeros((pattern_num,) + shape_2d, dtype=np.float64)
    photon_stack = np.zeros((pattern_num,) + shape_stack, dtype=np.float64)

    # ----------------------------------------------------
    # Loop through all the patterns
    # ----------------------------------------------------
    previous_run = run_list[0]
    # Get data source
    det, run, times, evt, info_dict = PsanaUtil.setup_exp(exp_name=exp_name,
                                                          run_num=previous_run,
                                                          det_name=det_name)
    tic = time.time()
    for lidx in range(pattern_num):  # local index

        # Get the current run number,
        # if the run number does not change,
        # then there is no need to create a new det object

        current_run = run_list[lidx]

        if current_run != previous_run:
            # Update the run number
            previous_run = current_run

            # Update the det object
            det, run, times, evt, info_dict = PsanaUtil.setup_exp(exp_name=exp_name,
                                                                  run_num=current_run,
                                                                  det_name=det_name)

        # Retireve the patterns
        adu_stack[lidx] = PsanaUtil.get_pattern_stack_fast(detector=det,
                                                           exp_run=run,
                                                           exp_times=times,
                                                           event_id=index_list[lidx])

        adu_2d[lidx] = PsanaUtil.get_pattern_2d_fast(detector=det,
                                                     exp_run=run,
                                                     exp_times=times,
                                                     event_id=index_list[lidx])

        photon_stack[lidx] = PsanaUtil.get_photon_stack_fast(detector=det,
                                                             exp_run=run,
                                                             exp_times=times,
                                                             event_id=index_list[lidx],
                                                             adu_per_photon=130)

        photon_2d[lidx] = PsanaUtil.get_photon_2d_fast(detector=det,
                                                       exp_run=run,
                                                       exp_times=times,
                                                       event_id=index_list[lidx],
                                                       adu_per_photon=130)

    h5_name = output_address + '{}_{}_{}.h5'.format(exp_name, tag, bidx)
    with h5.File(h5_name, 'w') as h5file:
        print("Save the data to {}".format(h5_name))
        # save_the pattern
        h5file.create_dataset('/batch_{}_index'.format(bidx), data=index_list)
        h5file.create_dataset('/batch_{}_run'.format(bidx), data=run_list)

        # h5file.create_dataset('/batch_{}_adu_stack'.format(bidx), data=adu_stack)
        # h5file.create_dataset('/batch_{}_adu_2d'.format(bidx), data=adu_2d)
        # h5file.create_dataset('/batch_{}_photon_stack'.format(bidx), data=photon_stack)
        h5file.create_dataset('/batch_{}_photon_2d'.format(bidx), data=photon_2d)

    toc = time.time()
    print("It takes {:.2f} seconds to process {} patterns.".format(toc - tic, pattern_num))
