import numpy as np


def get_scattered_momentum(position_in_meter, wavevector_in_meter):
    """
    This function calculate the corresponding momentum for each pixel on the detector.

    :param position_in_meter: Numpy array [pixel_num, 3]
                                    [ pixel_one-->[x,y,z],
                                      pixel_two-->[x,y,z],
                                      ...]

    :param wavevector_in_meter: Numpy array [x, y, z].
                                The wavevector is defined as 2pi / lambda (direction vector)
    :return: Numpy array [pixel_num, 3]
                    [ pixel_one-->[x,y,z],
                      pixel_two-->[x,y,z],
                        ...]
    """
    # Get the length and direction of the wavevector
    wavevector_length = np.linalg.norm(wavevector_in_meter)
    wavevector_direction = wavevector_in_meter / wavevector_length

    # Normalize the position vector
    pixel_distances = np.sqrt(np.sum(np.square(position_in_meter), axis=-1))
    pixel_directions = np.divide(position_in_meter, pixel_distances[:, np.newaxis])

    # Get the direction of the scattered momentum
    scattered_direction = pixel_directions - wavevector_direction[np.newaxis, :]

    # Scale the direction with the length
    scattered_momentum = wavevector_length * scattered_direction

    return scattered_momentum


def get_polarization_correction(pixel_position_m, reference_position_m, polarization):
    """
    This function calculate the polarization correction for each pixel

    This correction is

            |np.cross(pixel direction, polarization)|^2 =
                                        1 - np.square(np.dot(pixel direction, polarization))

    One should notice that this effect only applies for the intensity because I have
    introduced the square. Because if one consider the field, then the polarization
    factor should contain phase which can not be derived from this result.

    :param pixel_position_m: Pixel position in meter with respect to the origin.
                                [[x0, y0, z0]  --pixel0
                                 [x1, y1, z1]  --pixel1
                                 [x2, y2, z2]  --pixel2
                                 ...
                                 ]
    :param reference_position_m: The position of the interaction point w.r.t. to the origin.
    :param polarization: The polarization vector of the beam.
    :return:
    """
    # Get the relative position
    relative_position = pixel_position_m - reference_position_m[np.newaxis, :]

    # Calculate the distance and direction of each pixel in real space w.r.t. the reference point
    pixel_distances = np.sqrt(np.sum(np.square(relative_position), axis=-1))
    pixel_directions = np.divide(relative_position, pixel_distances[:, np.newaxis])

    # Calculate the correction
    polarization_correction = 1 - np.square(np.dot(pixel_directions, polarization))

    return polarization_correction


def get_thomson_factor():
    """
    This is the thomson factor when calculating the scattered field intensity.
    One only needs to multiply thi factor the the result calculated from atomic form factors.
    :return:
    """
    return 2.817895019671143 * 2.817895019671143 * 1e-30


###################################################################################################
# photon properties
###################################################################################################
def get_wavelength_m(photon_energy_ev):
    """
    Return the wavelength of the photon.

    :param: photon_energy: The photon energy in eV
    :return: 1.23984197386209e-06 / photon_energy . This is the wavelength in meter.
    """
    return 1.23984197386209e-06 / photon_energy_ev


def get_energy_ev(wavelength_m):
    """
    Return the energy of the photon.

    :param: wavelength_m: The wavelength energy in meter
    :return: 1.23984197386209e-06 / wavelength . This is the wavelength in meter.
            This is in eV
    """
    return 1.23984197386209e-06 / wavelength_m
