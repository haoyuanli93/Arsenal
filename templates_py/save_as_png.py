import sys
import os.path
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread as imr


threshold = 10


def main(argv):
    # Parse the parameters
    path_to_data = argv[1]
    path_to_save = argv[2]

    # Test if the folder and data exists
    if not (os.path.isfile(path_to_data)):
        print("The data file does not exist.")
        return 1

    if not (os.path.isfile(path_to_save)):
        print("The folder to save data does not exist.")
        print("Create the folder to save data.")
        subprocess.call(["mkdir %s" % path_to_save], shell=True)

    # Load the data
    # Note that the data has to be in the format [number of the data, dim1 of data, dim2 of data]
    data = np.load(path_to_data)
    num, height, width = data.shape

    # Convert the data to .jpg format
    for l in range(num):
        print("The %d th pattern" % l)
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w=1, h=1)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(data[l, :, :], vmax=threshold, aspect='normal')

        # Save the data. Notice that VGG16 only works for 224 by 224 images.
        fig.savefig(path_to_save + '/image_{}.png'.format('l'), dpi=224)
        plt.close()

    print("All the data are converted to the .jpg file and saved to %s folder" % path_to_save)

    del data

    # Now read all the image and save to the jpg_array variable
    jpg_array = np.zeros((num, 224, 224, 3))
    for l in range(num):
        print("The %d th pattern" % l)
        jpg = imr(path_to_save + '/' + str(l) + '.jpg')
        jpg_array[l, :, :, :] = jpg

    np.save(path_to_save + '/DataInJPGFormatArray.npy', jpg_array)
    # subprocess.call(["rm",path_to_save+"/*.jpg"], shell=True)

    print("Everything is set.")


if __name__ == "__main__":
    main(sys.argv)