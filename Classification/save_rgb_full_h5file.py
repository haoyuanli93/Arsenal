import matplotlib.pyplot as plt
import h5py as h5
import numpy as np
from imageio import imread
import sys
import time
import argparse

sys.path.append('/reg/neh/home5/haoyuan/Documents/my_repos/Arsenal')
import arsenal

"""
                        Notice

This is a special version of LCLS. In this version,
I has assumed the following facts.

1. One would like to process all the patterns in the h5file.
2. One do not care about the sequence       
    
"""

######################################################
# Define parameters
######################################################
# Parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--input_h5file', type=str, help=str("The h5file containing the patterns" +
                                                         " to render."))
parser.add_argument('--output_folder', type=str, help="The folder to save the patterns.")
parser.add_argument("--threshold", type=float, help="The threshold to render the pattern.")

# Parse
args = parser.parse_args()
input_h5file = args.input_h5file
output_folder = args.output_folder
threshold = args.threshold

######################################################
# Find out how many patterns are there in this h5file
######################################################

# Extract some basic information
with h5.File(input_h5file, 'r') as h5file:

    # Get batch number
    dataset_names = list(h5file.keys())

    # Get pattern number
    pattern_num = 0
    for dataset_name in dataset_names:
        dataset_shape = h5file[dataset_name].shape
        pattern_num += dataset_shape[0]

print("There are totally {} batches to process.".format(len(dataset_names)))
print("There are totally {} patterns to render.".format(pattern_num))

######################################################
# Begin the rendering
######################################################

# Check the output address.
arsenal.util.make_directory(path=output_folder)

# Open the h5 file.
with h5.File(input_h5file, 'r') as h5file:

    pattern_global_idx = 0
    for dataset_name in dataset_names:

        # Record the time
        tic = time.time()

        # Get the pattern
        pattern_set = np.array(h5file[dataset_name])

        # Loop through the patterns
        for pattern_local_idx in range(pattern_set.shape[0]):

            # Initialize a canvas without boundary
            fig = plt.figure(frameon=False)
            fig.set_size_inches(w=1, h=1)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            # Render the image
            ax.imshow(pattern_set[pattern_local_idx],
                      vmax=threshold,
                      aspect='auto', cmap='jet')

            # Save the data.
            # Notice that VGG16 only works for 224 by 224 images.
            # Notice that I need to name it with the global index in this h5file.
            fig.savefig(output_folder + '/image_{}.png'.format(pattern_global_idx), dpi=256)

            # Update the global index
            pattern_global_idx += 1

            # Close the figure
            plt.close(fig)

        # Record the time
        toc = time.time()
        print("It takes {:.2f} seconds to process the {} batch.".format(toc-tic,
                                                                        dataset_name))
    print("There are totally {} patterns in this h5file.".format(pattern_num))
    print("{} of the patterns are processed.".format(pattern_global_idx))

######################################################
# Load the patterns and save them to h5 file
#######################################################
# Now read all the image and save to the jpg_array variable
png_array = np.zeros((pattern_num, 256, 256, 3))
for l in range(pattern_num):
    png = imread(output_folder + '/image_{}.png'.format(l))
    png_array[l, :, :, :] = png[:, :, :3]

with h5.File(output_folder + '/png_format.h5', 'w') as h5file:
    h5file.create_dataset('patterns', data=png_array)
