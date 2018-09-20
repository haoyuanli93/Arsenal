import numpy as np
import h5py as h5
import time
import sys

import arsenal.psanautil

sys.path.append('/reg/neh/home5/haoyuan/Documents/my_repos/Arsenal')
import arsenal
from arsenal import lcls

###################################################################################
# Define parameters
###################################################################################
# Experiment info
exp_line = 'amo'
exp_name = 'amox26916'
user_name = 'haoyuan'
run_num = 85
det_name = 'pnccdFront'

# processing info
process_stage = 'scratch'

# Load mask
mask = np.load('/reg/d/psdm/amo/amox26916/scratch/haoyuan/psocake/r0085/masks/jet_streak_2.npy')

# Load the index to process
index_to_process = np.load('../output/classification_round_4_hits_global_idx.npy')

# Output info
output_address = '/reg/d/psdm/{}/{}/results/{}/'.format(exp_line, exp_name, user_name)

###################################################################################
# Initialize the datasource and detector and mask
###################################################################################
# Get data source
det, run, times, evt, info_dict = arsenal.psanautil.setup_exp(exp_name=exp_name,
                                                              run_num=run_num,
                                                              det_name=det_name)

# Get pattern number
pattern_num = index_to_process.shape[0]

# Get 2d boolean mask
mask_2d = det.image(evt=evt, nda_in=mask)
mask_bool = arsenal.util.cast_to_bool(mask=mask_2d, good=1, bad=0)

###################################################################################
# Loop through all patterns
###################################################################################
# Create holder for intensities
intensity_holder = np.zeros(pattern_num)

# holder for calculation time
time_holder = [0, ]
tic = time.time()

# Local counter
counter = 0
for pattern_idx in index_to_process:
    # Get the pattern
    sample = arsenal.psanautil.get_pattern_stack_fast(detector=det, exp_run=run, exp_times=times, event_id=pattern_idx)

    # Apply the mask
    sample_masked = sample[mask]

    # Get the intensity
    intensity_holder[counter] = np.sum(sample_masked)

    if np.mod(counter, 100) == 0:
        time_holder.append(time.time() - tic)
        print("{:.2f} seconds.".format(time_holder[-1]))

    # Update the counter
    counter += 1

# Save the result
output_file_name = output_address + '{}_run_{}_list_intensity_{}.h5'.format(exp_name,
                                                                                     run_num,
                                                                                     arsenal.util.time_stamp())
print("Processing results will be saved to folder {}.".format(output_file_name))
with h5.File(output_file_name, 'w') as h5file:
    # Exp info
    h5file.create_dataset(name="mask", data=mask)
    h5file.create_dataset(name="mask_2d", data=mask_2d)
    h5file.create_dataset(name='index list', data=index_to_process)
    # Result
    h5file.create_dataset(name="intensity", data=intensity_holder)
    # Calculation statistics
    h5file.create_dataset(name="Time step per 100 patterns", data=time_holder)
