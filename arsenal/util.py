import datetime
import os
import time

import numpy as np


def time_stamp():
    """
    Get a time stamp
    :return: A time stamp of the form '%Y_%m_%d_%H_%M_%S'
    """
    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    return stamp


###################################################################################################
# mask manipulation
###################################################################################################
def cast_to_bool(mask, good=1, bad=0):
    """
    Cast the mask from float or int to bool

    Value larger than (>=) (good + bad)/2 will be considered to be a good pattern.
    Value smaller than (<) (good + bad)/2 will be considered to be a bad pattern.

    :param mask: The mask array.
    :param good: The value for a good pixel.
    :param bad: The value for a bad pixel
    :return:
    """

    # Cast the mask to boolean values
    mask_bool = np.zeros_like(mask, dtype=np.bool)
    mask_bool[mask >= (good + bad) / 2.] = True
    mask_bool[mask < (good + bad) / 2.] = False

    return mask_bool


#################################################################################################
# File IO
#################################################################################################
def make_directory(path):
    """
    This function safely create a path. If a path already exists, it does nothing
    :param path:
    :return:
    """
    try:
        os.makedirs(path)
        print("A new folder is created at {}.".format(path))
    except OSError:
        if os.path.isdir(path):
            print("The folder {} already exists.".format(path))
        else:
            raise Exception("An error occurs when trying to create the folder {}".format(path) +
                            "Please check the folder yourself.")
