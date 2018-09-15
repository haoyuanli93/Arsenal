import time
import datetime


def time_stamp():
    """
    Get a time stamp
    :return: A time stamp of the form '%Y_%m_%d_%H_%M_%S'
    """
    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    return stamp


###########################################################################################################
# properties
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
