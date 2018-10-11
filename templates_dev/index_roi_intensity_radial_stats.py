import sys
import numpy as np
import h5py as h5
import time

sys.path.append('/reg/neh/home5/haoyuan/Documents/my_repos/Arsenal')
import arsenal
from arsenal import PsanaUtil


#######################################################################################################################
# Define parameters
#######################################################################################################################
exp_line = 'amo'
exp_name = 'amox34117'
user_name = 'haoyuan'
run_num = 176
det_name = 'pnccdFront'
process_stage = 'scratch'

# Specify the output address
output_address = '/reg/d/psdm/{}/{}/results/{}/'.format(exp_line, exp_name, user_name)

# Load mask
mask_2d_bool_roi = np.load('../../output/mask_2d_bool_roi_round_1.npy')

# Category map 2D roi masked
catmap_2d_roi_masked = np.load('../../output/catmap_2d_roi_masked_round_1.npy')

# Get event index energy
index_to_process = arsenal.lcls.get_cxi_pattern_idx(exp_line=exp_line,
                                     exp_name=exp_name,
                                     user_name=user_name,
                                     process_stage=process_stage,
                                     run_num=run_num)

# Define roi
roi = [[362, 646], [369, 513]]

#######################################################################################################################
# Initialize datasource and the detector
#######################################################################################################################
# Get data source
det, run, times, evt, info_dict = PsanaUtil.setup_exp(exp_name=exp_name,
                                                 run_num=run_num,
                                                 det_name=det_name)

# Get pattern number
pattern_num = index_to_process.shape[0]

#######################################################################################################################
# Get masked category map
#######################################################################################################################
category_list = np.sort(np.unique(catmap_2d_roi_masked))
category_num = category_list.shape[0]
print("There are {} categories to calculate.".format(category_num))
print(category_list)

#######################################################################################################################
# Loop through all patterns
#######################################################################################################################
# Create holder for intensities
intensity_holder = np.zeros(pattern_num)
# Create holder for mean
mean_holder = np.zeros((pattern_num, category_num))
# Create holder for std
std_holder = np.zeros((pattern_num, category_num))
# Create holder for max
max_holder = np.zeros((pattern_num, category_num))
# Create holder for min
min_holder = np.zeros((pattern_num, category_num))

# holder for calculation time
time_holder = [0, ]
tic = time.time()

# Create a local counter
counter = 0
for pattern_idx in index_to_process:
    # Get the pattern
    sample = PsanaUtil.get_photon_2d_fast(detector=det, exp_run=run,
                              exp_times=times,
                              event_id=pattern_idx,
                              adu_per_photon=130)
    
    # Extract the roi region of the sample
    sample_roi = sample[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
    
    # Apply the mask
    sample_roi_masked = sample_roi[mask_2d_bool_roi]

    # Get the intensity
    intensity_holder[counter] = np.sum(sample_roi_masked)

    # Get the distribution
    for cat_idx in range(category_num):
        mean_holder[counter, cat_idx] = np.mean(sample_roi_masked[catmap_2d_roi_masked == category_list[cat_idx]])
        std_holder[counter, cat_idx] = np.std(sample_roi_masked[catmap_2d_roi_masked == category_list[cat_idx]])
        min_holder[counter, cat_idx] = np.min(sample_roi_masked[catmap_2d_roi_masked == category_list[cat_idx]])
        max_holder[counter, cat_idx] = np.max(sample_roi_masked[catmap_2d_roi_masked == category_list[cat_idx]])

    if np.mod(counter, 100) == 0:
        time_holder.append(time.time() - tic)
        print("{:.2f} seconds.".format(time_holder[-1]))

    # update the counter
    counter += 1

# Get a time stamp
output_file = output_address + 'radial_distribution_and_intensity_run_{}_all_{}.h5'.format(run_num,
                                                                                           arsenal.util.time_stamp())
print("Processing results will be saved to folder {}.".format(output_file))

# Save the result
with h5.File(output_file, 'w') as h5file:
    # Exp info
    h5file.create_dataset(name="mask", data=mask_2d_bool_roi)
    h5file.create_dataset(name="run_num", data=run_num)
    h5file.create_dataset(name="roi", data=np.array(roi))

    # Category info
    h5file.create_dataset(name="category_map", data=catmap_2d_roi_masked)
    h5file.create_dataset(name="category_num", data=category_num)
    h5file.create_dataset(name="category_list", data=category_list)

    # Radial distribution
    h5file.create_dataset(name="radial_mean", data=mean_holder)
    h5file.create_dataset(name="radial_std", data=std_holder)
    h5file.create_dataset(name="radial_min", data=min_holder)
    h5file.create_dataset(name="radial_max", data=max_holder)
    h5file.create_dataset(name="intensity", data=intensity_holder)

    # Average of the distribution
    h5file.create_dataset(name="average_radial_mean", data=np.mean(mean_holder, axis=0))
    h5file.create_dataset(name="average_radial_std", data=np.mean(std_holder, axis=0))
    h5file.create_dataset(name="average_radial_min", data=np.mean(min_holder, axis=0))
    h5file.create_dataset(name="average_radial_max", data=np.mean(max_holder, axis=0))

    # Calculate the name
    h5file.create_dataset(name="Time step per 100 patterns", data=time_holder)
