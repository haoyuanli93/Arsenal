import time
import datetime
import numpy as np


def time_stamp():
    """
    Get a time stamp
    :return: A time stamp of the form '%Y_%m_%d_%H_%M_%S'
    """
    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    return stamp


###########################################################################################################
# photon properties
###########################################################################################################
def get_wavelength(photon_energy):
    """
    Return the wavelength of the photon.

    :param: photon_energy: The photon energy in eV
    :return: 1.23984197386209e-06 / photon_energy . This is the wavelength in meter.
    """
    return 1.23984197386209e-06 / photon_energy


def get_energy(wavelength):
    """
    Return the energy of the photon.

    :param: wavelength: The wavelength energy in meter
    :return: 1.23984197386209e-06 / wavelength . This is the wavelength in meter.
    """
    return 1.23984197386209e-06 / wavelength


###########################################################################################################
# mask manipulation
###########################################################################################################
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
    mask_bool[mask >= (good + bad) / 2] = True
    mask_bool[mask < (good + bad) / 2] = False

    return mask_bool
