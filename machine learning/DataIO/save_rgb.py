import matplotlib.pyplot as plt
import h5py as h5
import numpy as np
from imageio import imread
import sys
import time

sys.path.append('/reg/neh/home5/haoyuan/Documents/my_repos/Arsenal')
import arsenal

######################################################
# Define parameters
######################################################
run_num = 227
threshold = 4
path_to_save = '../output/run{}'.format(run_num)

# Load data
with h5.File('../../hitfinder_v2/output/r{}/step_1_intensity_and_patterns.h5'.format(run_num), 'r') as h5file:
    patterns = np.array(h5file['patterns'][:, :224, :])

######################################################
# Process the data and check the path
######################################################
arsenal.util.make_directory(path=path_to_save)
pattern_num = patterns.shape[0]
padded_patterns = np.pad(array=patterns,
                         pad_width=((0, 0), (0, 0), (0, 224 - patterns.shape[-1])),
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
    fig.savefig(path_to_save + '/image_{}.png'.format(l), dpi=224)

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
    png = imread(path_to_save + '/image_{}.png'.format(l))
    png_array[l, :, :, :] = png[:, :, :3]

with h5.File('../output/run{}/png_format.h5'.format(run_num), 'w') as h5file:
    h5file.create_dataset('patterns', data=png_array)
