import sys
import numpy as np
import h5py as h5
import time 
import datetime
import psana

import arsenal.util

sys.path.append('/reg/neh/home5/haoyuan/Documents/my_repos/Arsenal')
import arsenal
import arsenal.lcls

#######################################################################################################################
# Define parameters
#######################################################################################################################
exp_line = 'amo'
exp_name = 'amox26916'
user_name = 'haoyuan'

process_stage = 'scratch'

run_num = 85
det_name = 'pnccdFront'

output_address = '/reg/d/psdm/{}/{}/results/{}/'.format(exp_line, exp_name,
                                                       user_name)

# Construct the file address of the corresponding cxi file
file_name = '/reg/d/psdm/{}/{}/{}/{}/psocake/r{:0>4d}/{}_{:0>4d}.cxi'.format(exp_line, exp_name, process_stage,
                                                    user_name, run_num, exp_name, run_num)
print("The cxi file is located at {}".format(file_name))
print("Processing results will be saved to folder {}.".format(output_address))

#######################################################################################################################
# Initialize datasource and the detector
#######################################################################################################################
# Get data source
ds = psana.DataSource('exp={}:run={}:idx'.format(exp_name, run_num))
run = ds.runs().next()
env = ds.env()
times = run.times()
evt = run.event(times[0])

# Get pattern number
pattern_num = len(times)
print("There are {} patterns in this run in total.".format(pattern_num))

# Get photon energy
with  h5.File(file_name, 'r') as h5file:
    holder = h5file['/LCLS/photon_wavelength_A'].value
    # convert to meter
    photon_wavelength = holder[0] / (10 ** 10)
    photon_energy = arsenal.util.get_energy(wavelength=photon_wavelength)
print("The photon wave length is {} m.".format(photon_wavelength))

# Get detector
det = psana.Detector('pnccdFront', env)

#######################################################################################################################
# Load mask and cast to bool
#######################################################################################################################
# Load mask
mask = np.load('/reg/d/psdm/amo/amox26916/scratch/haoyuan/psocake/r0085/masks/jet_streak.npy')

# Get 2d mask
mask_2d = det.image(nda_in=mask, evt=evt)

# Cast the mask to boolean values
mask_bool = np.zeros_like(mask, dtype=np.bool)
mask_bool[mask > 0.5] = True
mask_bool[mask < 0.5] = False

#######################################################################################################################
# Get momentum mesh
#######################################################################################################################
# Visualization Parameters
number_of_interval = 300
radial_range = "auto"

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
category_map_masked = category_map[mask == True]
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
time_holder = [0,]
tic = time.time()

for pattern_idx in range(pattern_num):
    # Get the pattern
    sample = arsenal.lcls.get_pattern_stack(detector=det, exp_run=run, event_id=pattern_idx)
    
    # Apply the mask
    sample_masked = sample[mask==True]
    
    # Get the distribution
    for cat_idx in range(category_num):
        holder[pattern_idx, cat_idx] = np.mean(sample_masked[category_map_masked == category_list[cat_idx]])
    
    intensity_holder[pattern_idx] = np.sum(sample_masked)
    
    if np.mod(pattern_idx, 100) == 0:
        time_holder.append(time.time() - tic)
        print("{:.2f} seconds.".format(time_holder[-1]))
    
# Get a time stamp
stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

# Save the result
with h5.File(output_address + 'radial_distribution_{}.h5'.format(stamp), 'w') as h5file:
    h5file.create_dataset(name="mask", data=mask)
    h5file.create_dataset(name="category_map", data=category_map)
    h5file.create_dataset(name="run_num", data=run_num)
    h5file.create_dataset(name="radial_distribution", data=holder)
    h5file.create_dataset(name="category_num", data=category_num)
    h5file.create_dataset(name="category_list", data=category_list)
    h5file.create_dataset(name="intensity", data=intensity_holder)
