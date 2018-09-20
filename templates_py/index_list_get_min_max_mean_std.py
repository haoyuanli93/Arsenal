import sys
import numpy as np
import h5py as h5
import time

import arsenal.PsanaUtil

sys.path.append('/reg/neh/home5/haoyuan/Documents/my_repos/Arsenal')
import arsenal
from arsenal import lcls

################################################################################
# Define parameters
################################################################################
exp_line = 'amo'
exp_name = 'amox26916'
user_name = 'haoyuan'
process_stage = 'scratch'
run_num = 85
det_name = 'pnccdFront'

# get a tag
info = 'hits'
tag = 'hits'

# The numpy array file containing indexes to inspect
index_to_process = np.load('../output/classification_round_4_hits_global_idx.npy')

# Specify the output address
output_address = '../output/'

################################################################################
# Intialize the detector
################################################################################
# Get data source
det, run, times, evt, info_dict = arsenal.PsanaUtil.setup_exp(exp_name=exp_name,
                                                              run_num=run_num,
                                                              det_name=det_name)

# Get pattern number
pattern_num = index_to_process.shape[0]

################################################################################
# Divide the index list
################################################################################
sub_lists_num = index_to_process.shape[0] // 50
print("There are roughly {} batches to process.".format(sub_lists_num))

# Get sublists 
sub_lists = np.array_split(ary=index_to_process, indices_or_sections=sub_lists_num, axis=0)

################################################################################
# Calculate the statistics
################################################################################
# For the first sublist
data_holder = np.zeros((sub_lists[0].shape[0], 4, 512, 512))

tic = time.time()
counter = 0
for idx in sub_lists[0]:
    # Get the pattern
    data_holder[counter] = arsenal.PsanaUtil.get_pattern_stack(detector=det, exp_run=run, event_id=idx)
    # Update the local index
    counter += 1

toc = time.time()
print("It takes {:.2f} seconds to process {} patterns.".format(toc - tic, sub_lists[0].shape[0]))

# Process the first batch
min_holder = np.min(data_holder, axis=0)
max_holder = np.max(data_holder, axis=0)
sum_holder = np.sum(data_holder, axis=0)
square_sum_holder = np.sum(np.square(data_holder), axis=0)

################################################################################
# Process the other batches
################################################################################
# Get a counter for different batches
batch_counter = 1

for sub_list in sub_lists[1:]:

    # For the specific sublist
    data_holder = np.zeros((sub_list.shape[0], 4, 512, 512))

    tic = time.time()
    counter = 0
    for idx in sub_list:
        # Get the pattern
        data_holder[counter] = arsenal.PsanaUtil.get_pattern_stack(detector=det, exp_run=run, event_id=idx)
        # Update the local index
        counter += 1

    toc = time.time()
    print("It takes {:.2f} seconds to process {} patterns.".format(toc - tic, sub_lists[0].shape[0]))

    # Get statistics
    tmp_min_holder = np.min(data_holder, axis=0)
    tmp_max_holder = np.max(data_holder, axis=0)
    tmp_sum_holder = np.sum(data_holder, axis=0)
    tmp_square_sum_holder = np.sum(np.square(data_holder), axis=0)

    # Update the previous results
    min_holder = np.minimum(min_holder, tmp_min_holder)
    max_holder = np.maximum(max_holder, tmp_max_holder)
    sum_holder += tmp_sum_holder
    square_sum_holder += tmp_square_sum_holder

    # Update the batch_counter
    print("Finishes processing batch {}".format(batch_counter))
    batch_counter += 1

# Convert the sum and square sum into mean and std
mean_holder = sum_holder / float(pattern_num)
std_holder = np.sqrt(square_sum_holder / pattern_num - np.square(mean_holder))
std_holder *= np.sqrt(pattern_num) / np.sqrt(pattern_num - 1)

# Get a time stamp
stamp = arsenal.util.time_stamp()

# Save the result
with h5.File(output_address + 'statistics_{}_{}_{}_{}.h5'.format(tag, exp_name, run_num, stamp), 'w') as h5file:

    # Save some parameters
    h5file.create_dataset("pattern_num", data=pattern_num)

    # Save pattern stack
    h5file.create_dataset("min_stack", data=min_holder)
    h5file.create_dataset("max_stack", data=max_holder)
    h5file.create_dataset("sum_stack", data=sum_holder)
    h5file.create_dataset("square_sum_stack", data=square_sum_holder)
    h5file.create_dataset("mean_stack", data=mean_holder)
    h5file.create_dataset("std_stack", data=std_holder)

    # Turn all the stack image into 2D patterns
    h5file.create_dataset("min_2d", data=det.image(nda_in=min_holder, evt=evt))
    h5file.create_dataset("max_2d", data=det.image(nda_in=max_holder, evt=evt))
    h5file.create_dataset("sum_2d", data=det.image(nda_in=sum_holder, evt=evt))
    h5file.create_dataset("square_sum_2d", data=det.image(nda_in=square_sum_holder, evt=evt))
    h5file.create_dataset("mean_2d", data=det.image(nda_in=mean_holder, evt=evt))
    h5file.create_dataset("std_2d", data=det.image(nda_in=std_holder, evt=evt))
