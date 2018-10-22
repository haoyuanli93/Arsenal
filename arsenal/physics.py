import numpy as np


def get_scattered_momentum(position_in_meter, wavevector_in_meter):
    """
    This function calculate the corresponding momentum for each pixel on the detector.

    :param position_in_meter: Numpy array [pixel_num, 3]
                                    [ pixel_one-->[x,y,z],
                                      pixel_two-->[x,y,z],
                                      ...]

    :param wavevector_in_meter: Numpy array [x, y, z]. The wavevector is defined as 2pi / lambda (direction vector)
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

    # Scale the direction with the lenght
    scattered_momentum = wavevector_length * scattered_direction

    return scattered_momentum


def get_polarization_effect(pixel_position_m, pixel_orientation, wavevector_in_meter):
    """
    
    :param pixel_position_m:
    :param pixel_orientation:
    :param wavevector_in_meter:
    :return:
    """
    pass
