import matplotlib.pyplot as plt
import h5py as h5
import numpy as np
from imageio import imread
import sys
import time
import argparse

sys.path.append('/reg/neh/home5/haoyuan/Documents/my_repos/Arsenal')
import arsenal

######################################################
# Define parameters
######################################################
# Parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--input_h5file', type=str, help="The h5file containing the patterns to render.")
parser.add_argument('--output_folder', type=str, help="The folder to save the patterns.")
parser.add_argument("--dataset_name", type=str, help="The dataset in the h5file containing the patterns to render.")
parser.add_argument("--threshold", type=float, help="The threshold to render the pattern.")

# Parse
args = parser.parse_args()
input_h5file = args.input_h5file
dataset_name = args.dataset_name
output_folder = args.output_folder
threshold = args.threshold

# Load data
with h5.File(input_h5file, 'r') as h5file:
    patterns = np.array(h5file[dataset_name])

######################################################
# Process the data and check the path
######################################################
arsenal.util.make_directory(path=output_folder)
pattern_num = patterns.shape[0]

if patterns.shape[1] > patterns.shape[2]:

    padded_patterns = np.pad(array=patterns,
                             pad_width=((0, 0), (0, 0), (0, patterns.shape[1] - patterns.shape[2])),
                             mode='constant',
                             constant_values=0)
else:
    padded_patterns = np.pad(array=patterns,
                             pad_width=((0, 0), (0, patterns.shape[2] - patterns.shape[1]), (0, 0)),
                             mode='constant',
                             constant_values=0)

######################################################
# Show pattern without boundary and any other stuff
#######################################################
tic = time.time()
for l in range(pattern_num):
    # Initialize a canvas without boundary
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w=1, h=1)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Render the image
    ax.imshow(padded_patterns[l], vmax=threshold, aspect='auto', cmap='jet')

    # Save the data. Notice that VGG16 only works for 224 by 224 images.
    fig.savefig(output_folder + '/image_{}.png'.format(l), dpi=224)

    # Close the figure
    plt.close(fig)
toc = time.time()

print('It takes {:.2f} seconds to render {} patterns.'.format(toc - tic, pattern_num))

######################################################
# Load the patterns and save them to h5 file
#######################################################
# Now read all the image and save to the jpg_array variable
png_array = np.zeros((pattern_num, 224, 224, 3))
for l in range(pattern_num):
    png = imread(output_folder + '/image_{}.png'.format(l))
    png_array[l, :, :, :] = png[:, :, :3]

with h5.File(output_folder + '/png_format.h5', 'w') as h5file:
    h5file.create_dataset('patterns', data=png_array)
