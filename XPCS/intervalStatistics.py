import numpy as np
import h5py as h5


########################################################################################
#     Define functions
########################################################################################
# Reconstruct the image from the photon distribution
def reconstruct_img(photons_x, photons_y, shape):
    nx, ny = shape
    phot_img, _, _ = np.histogram2d(photons_y + 0.5, photons_x + 0.5, bins=[np.arange(nx + 1), np.arange(ny + 1)])
    return phot_img


########################################################################################
#     Specify the calculation details
########################################################################################
run_num = 152

# Load a run
with h5.File("/reg/d/psdm/xpp/xppc00120/results/smalldata_output_hy/" +
             "xppc00120_Run0{}.h5".format(run_num)) as datafile:
    # Load the intensity level indicator
    diode2_holder = np.array(datafile['diode2/channels'])
    diodeU_holder = np.array(datafile['diodeU/channels'])
    ipm2_holder = np.array(datafile['ipm2/sum'])
    l3e_holder = np.array(datafile['ebeam/L3_energy'])

    # Load the cc/vcc branches indicator
    cc_data_holder = np.array(datafile['ai/ch06'])
    vcc_data_holder = np.array(datafile['ai/ch07'])

    photon_x_holder = []
    photon_y_holder = []
    # Loop though the epix detectors to get the photons
    for epix_idx in range(1, 5):
        photon_x_holder.append(np.array(datafile['epix_alc{}/ragged_droplet_photon_i'.format(epix_idx)]))
        photon_y_holder.append(np.array(datafile['epix_alc{}/ragged_droplet_photon_j'.format(epix_idx)]))

# Get photon interval statistics
pattern_num = cc_data_holder.shape[0]

# Get a interval holder
intervalStatistics = [[[[] for y in range(768)] for x in range(704)] for ePix_idx in range(4)]

# Loop through all data
for pattern_idx in range(pattern_num):
    for ePix_idx in range(4):
        # Get the unique values for the pixels
        xValue, xValueIndex = np.unique(photon_x_holder[ePix_idx], return_index=True)
        yValue, yValueIndex = np.unique(photon_y_holder[ePix_idx], return_index=True)

        # Add the pattern index to the intervalStatistics holder
        intervalStatistics[ePix_idx][xValue[np.argsort(xValueIndex)]][yValue[np.argsort(yValueIndex)]].append(pattern_idx)
