import sys
import argparse
import numpy as np
import h5py as h5
import time 
import datetime
import matplotlib.pyplot as plt

import psana

sys.path.append('/reg/neh/home5/haoyuan/Documents/my_repos/Arsenal')
import arsenal
import arsenal.lcls

################################################################################
# Specify the parameters to use
################################################################################
exp_line = 'amo'
exp_name = 'amox26916'
user_name = 'haoyuan'

#get a tag
tag = 'hits'

process_stage = 'scratch'

run_num = 85
det_name = 'pnccdFront'

# Construct the file address of the corresponding cxi file
file_name = '/reg/d/psdm/{}/{}/{}/{}/psocake/r{:0>4d}/{}_{:0>4d}.cxi'.format(exp_line,exp_name,process_stage,
                                                    user_name,run_num,exp_name,run_num)
print("The cxi file is located at {}".format(file_name))

# The numpy array file containing indexes to inspect
index_to_process = np.load('../output/classification_round_4_hits_global_idx.npy')

# Get pattern number
pattern_num = len(index_to_process)
print("There are {} patterns in this run in total.".format(pattern_num))

# Specify the output address
output_address = '../output/'
print("Processing results will be saved to folder {}.".format(output_address))
################################################################################
# Intialize the detector
################################################################################
# Get data source
ds = psana.DataSource('exp={}:run={}:idx'.format(exp_name, run_num))
run = ds.runs().next()    
env = ds.env()
times = run.times()
evt = run.event(times[0])

# Get photon energy
with  h5.File(file_name, 'r') as h5file:
    holder = h5file['/LCLS/photon_wavelength_A'].value
    # convert to meter
    photon_wavelength = holder[0] / (10**10)
    photon_energy = arsenal.radial.get_energy(wavelength=photon_wavelength)
print("The photon wave length is {} m.".format(photon_wavelength))

# Get detector
det = psana.Detector('pnccdFront', env)

################################################################################
# Divide the index list
################################################################################
sub_lists_num = index_to_process.shape[0] // 50
print("There are roughly {} batches to process.".format(sub_lists_num))

# Get sublists 
sub_lists = np.array_split(ary=index_to_process,indices_or_sections=sub_lists_num, axis=0)

################################################################################
# Calculate the statistics
################################################################################
# For the first sublist
data_holder = np.zeros((sub_lists[0].shape[0], 4, 512, 512))

tic = time.time()
counter = 0
for idx in sub_lists[0]:
    
    # Get the pattern
    data_holder[counter] = arsenal.lcls.get_pattern_stack(detector=det, exp_run=run, event_id=idx)
    # Update the local index
    counter += 1
    
toc = time.time()
print("It takes {:.2f} seconds to process {} patterns.".format(toc-tic, sub_lists[0].shape[0]))

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
        data_holder[counter] = arsenal.lcls.get_pattern_stack(detector=det, exp_run=run, event_id=idx)
        # Update the local index
        counter += 1

    toc = time.time()
    print("It takes {:.2f} seconds to process {} patterns.".format(toc-tic, sub_lists[0].shape[0]))
    
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

# Get a time stamp
stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    
# Save the result
with h5.File(output_address + 'statistics_{}_{}_{}_{}.h5'.format(tag, exp_name, run_num, stamp), 'w') as  h5file:
    h5file.create_dataset("min_stack", data=min_holder)
    h5file.create_dataset("max_stack", data=max_holder)
    h5file.create_dataset("sum_stack", data=sum_holder)
    h5file.create_dataset("square_sum_stack", data=square_sum_holder)
    
    # Get mean and std
    mean_holder = sum_holder / float(pattern_num)
    std_holder = np.sqrt(square_sum_holder / pattern_num - np.square(mean_holder))
    std_holder *= np.sqrt(pattern_num) / np.sqrt(pattern_num - 1)
    
    # Save the results
    h5file.create_dataset("pattern_num", data= pattern_num)
    h5file.create_dataset("mean_stack", data=mean_holder)
    h5file.create_dataset("std_stack", data=std_holder)
    
    # Turn all the stack image into 2D patterns
    h5file.create_dataset("min_2d", data=det.image(nda_in=min_holder, evt=evt))
    h5file.create_dataset("max_2d", data=det.image(nda_in=max_holder, evt=evt))
    h5file.create_dataset("sum_2d", data=det.image(nda_in=sum_holder, evt=evt))
    h5file.create_dataset("square_sum_2d", data=det.image(nda_in=square_sum_holder, evt=evt))
    h5file.create_dataset("mean_2d", data=det.image(nda_in=mean_holder, evt=evt))
    h5file.create_dataset("std_2d", data=det.image(nda_in=std_holder, evt=evt))