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
exp_name = 'amox26916'
user_name = 'haoyuan'
run_num = 85
det_name = 'pnccdFront'
process_stage = 'scratch'

# Specify the output address
output_address = '/reg/d/psdm/{}/{}/results/{}/'.format(exp_line, exp_name, user_name)

# Load mask
mask = np.load('/reg/d/psdm/amo/amox26916/scratch/haoyuan/psocake/r0085/masks/pnccdFront_mask_hit_finder.npy')

# Load the index to process
index_to_process = np.load('../output/classification_round_4_hits_global_idx.npy')

# Visualization Parameters
number_of_interval = 300
radial_range = "auto"

#######################################################################################################################
# Initialize datasource and the detector
#######################################################################################################################
# Get data source
det, run, times, evt, info_dict = PsanaUtil.setup_exp(exp_name=exp_name,
                                                      run_num=run_num,
                                                      det_name=det_name)

# Get pattern number
pattern_num = index_to_process.shape[0]

# Get photon energy
photon_energy = arsenal.lcls.get_cxi_photon_energy(exp_line=exp_line,
                                                   exp_name=exp_name,
                                                   process_stage=process_stage,
                                                   user_name=user_name,
                                                   run_num=run_num)

#######################################################################################################################
# Load mask and cast to bool
#######################################################################################################################
# Get 2d mask
mask_2d = det.image(nda_in=mask, evt=evt)

# Cast the mask to boolean values
mask_bool = arsenal.util.cast_to_bool(mask=mask, good=1, bad=0)

#######################################################################################################################
# Get momentum mesh
#######################################################################################################################
# Initialize
(category_map,
 momentum_length_map,
 momentum_steps,
 p_correction,
 g_correction) = arsenal.radial.wrapper_get_pixel_map(detector=det,
                                                      run_num=run_num,
                                                      photon_energy=photon_energy,
                                                      number_of_interval=number_of_interval,
                                                      radial_range=radial_range)

#######################################################################################################################
# Get masked category map
#######################################################################################################################
category_map_masked = category_map[mask]
category_list = np.sort(np.unique(category_map_masked))
category_num = category_list.shape[0]
print("There are {} categories to calculate.".format(category_num))
print(category_list)

#######################################################################################################################
# Loop through all patterns
#######################################################################################################################
# Create a holder for all the distributions
holder = np.zeros((pattern_num, category_num))
# Create holder for intensities
intensity_holder = np.zeros(pattern_num)

# holder for calculation time
time_holder = [0, ]
tic = time.time()

# Create a local counter
counter = 0
for pattern_idx in index_to_process:
    # Get the pattern
    sample = PsanaUtil.get_pattern_stack_fast(detector=det, exp_run=run, exp_times=times, event_id=pattern_idx)

    # Apply the mask
    sample_masked = sample[mask]

    # Get the intensity
    intensity_holder[counter] = np.sum(sample_masked)

    # Get the distribution
    for cat_idx in range(category_num):
        holder[counter, cat_idx] = np.mean(sample_masked[category_map_masked == category_list[cat_idx]])

    if np.mod(counter, 100) == 0:
        time_holder.append(time.time() - tic)
        print("{:.2f} seconds.".format(time_holder[-1]))

    # update the counter
    counter += 1

# Get a time stamp
output_file = output_address + 'radial_distribution_and_intensity_run_{}_list_{}.h5'.format(run_num,
                                                                                            arsenal.util.time_stamp())
print("Processing results will be saved to folder {}.".format(output_file))

# Save the result
with h5.File(output_file, 'w') as h5file:
    # Exp info
    h5file.create_dataset(name="mask", data=mask)
    h5file.create_dataset(name="mask 2d", data=mask_2d)
    h5file.create_dataset(name="run_num", data=run_num)
    h5file.create_dataset(name='index list', data=index_to_process)

    # Category info
    h5file.create_dataset(name="category_map", data=category_map)
    h5file.create_dataset(name="category_num", data=category_num)
    h5file.create_dataset(name="category_list", data=category_list)

    # Radial distribution
    h5file.create_dataset(name="radial_distribution", data=holder)
    h5file.create_dataset(name="intensity", data=intensity_holder)

    # Average of the distribution
    h5file.create_dataset(name="average_radial_distribution", data=np.mean(holder, axis=0))

    # Calculate the name
    h5file.create_dataset(name="Time step per 100 patterns", data=time_holder)
