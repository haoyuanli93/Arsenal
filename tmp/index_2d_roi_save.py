import sys
import os
import numpy as np
import h5py as h5
import time

sys.path.append('/reg/neh/home5/haoyuan/Documents/my_repos/Arsenal')
import arsenal
from arsenal import PsanaUtil


####################################################################################################
# [USER] Specify the parameters to use
####################################################################################################
# The experiment info
exp_line = 'amo'
exp_name = 'amox34117'
user_name = 'haoyuan'
process_stage = 'scratch'
run_num = 182
det_name = 'pnccdFront'

# Add a tag
tag = 'selection_based_on_psocake'

# Construct the output address
output_address = '/reg/d/psdm/{}/{}/{}/{}/experiment_data/'.format(exp_line, exp_name, process_stage, user_name)

# Construct the output file name 
output_name = '{}_run_{}_{}.h5'.format(exp_name, run_num, tag)

# Get event index energy
index_to_process = arsenal.lcls.get_cxi_pattern_idx(exp_line=exp_line,
                                     exp_name=exp_name,
                                     user_name=user_name,
                                     process_stage=process_stage,
                                     run_num=run_num)

# Define roi
roi = [[385, 625], [390, 513]]


####################################################################################################
# [AUTO] Check the parameters
####################################################################################################
if not os.path.isdir(output_address):
    os.mkdir(output_address)
print("The output address is {}".format(output_address))

####################################################################################################
# [AUTO] Intialize the detector
####################################################################################################
# Get data source
det, run, times, evt, info_dict = PsanaUtil.setup_exp(exp_name=exp_name,
                                     run_num=run_num,
                                     det_name=det_name)

# Get pattern number
pattern_num = index_to_process.shape[0]


####################################################################################################
# [AUTO] Divide the index list
####################################################################################################
if pattern_num <= 100:
    sub_lists = [index_to_process, ]
else:
    sub_lists_num = pattern_num // 100
    print("There are roughly {} batches to process.".format(sub_lists_num))
    # Get sublists 
    sub_lists = np.array_split(ary=index_to_process, indices_or_sections=sub_lists_num, axis=0)

####################################################################################################
# [AUTO] Load and downsample all the patterns
####################################################################################################
with h5.File(output_address + output_name, 'w') as h5file:
    # For different batches
    batch_counter = 0

    # Loop through this list of sublists
    for sublist in sub_lists:

        tic = time.time()
        # First, get to know the index number in this list
        idx_num = sublist.shape[0]
        # Construct the shape of the holder variable
        holder_shape = (idx_num, roi[0][1] - roi[0][0], roi[1][1] - roi[1][0])
        holder = np.zeros(holder_shape)

        # Extract all the patterns from this sublist
        idx_counter = 0
        for idx in sublist:
            sample = PsanaUtil.get_photon_2d_fast(detector=det, 
                                      exp_run=run,
                                      exp_times=times,
                                      event_id=idx,
                                      adu_per_photon=130)
            
            sample_roi = sample[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
            
            holder[idx_counter] = sample_roi
            idx_counter += 1

        # save_the down sampled pattern
        h5file.create_dataset('/batch_{}_index'.format(batch_counter), data=sublist)
        h5file.create_dataset('/batch_{}_pattern'.format(batch_counter),data=holder)

        # Update the batch_counter
        batch_counter += 1

        toc = time.time()
        print("It takes {:.2f} seconds to process {} patterns.".format(toc - tic, idx_num))
