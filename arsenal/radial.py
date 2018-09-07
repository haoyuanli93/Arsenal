import numpy as np


def get_radial_distribution(pattern, category_map, number_of_interval):
    """
    Return the radial distribution based on the category_map. 
    :param: pattern: This is the dataset to inspect. Notice that this
                    has to be the pattern stack obtained from det.calib rather than det.image.
    :param: category_map : The category map of the pixel.
    :param: number_of_interval: The number of intervals.
    :return: A numpy array containing the radial distriabution. The shape is (number_of_interval,)
    """
    distribution = np.zeros(number_of_interval)
    # Calculate the distribution
    for category_idx in range(number_of_interval):
        distribution[category_idx] = np.mean(pattern[category_map == category_idx])

    return distribution


def wrapper_get_pixel_map(detector, run_num,
                          photon_energy, number_of_interval, radial_range="auto"):
    """
    This wrapper function takes the detector instance and the number of intervals and the radial arange to inspect.

    :param: detector : The detector instance
    :param: number_of_intervals: The number of intervals to dividies the radial range.
    :param: radial_range: The radial range to inspect. Notice that this is the momentum range to inspect.
         The unit is 1/meter. The way to calculate the momentum is 2*np.pi/wavelength. The wavelength is in meter.
         This function also return a momentum length_map, you can modify your choice with knowledge from that
         output.

    :return: category map, momentum_length_map
    """

    # Get momentum length map stack
    coordinate = detector.coords_xyz(par=run_num)
    momentum_map_stack, coordinate_new, direction, distance = get_momentum_map(coor_xyz=coordinate,
                                                                               photon_energy=photon_energy)
    momentum_length_map_stack = np.sqrt(np.sum(np.square(momentum_map_stack), axis=-1))

    # Detector distance
    distance_detector = np.mean(coordinate[2])

    # Get ends for radial distribution
    if radial_range == "auto":
        # Set the ends for each category
        momentum_length_max = np.max(momentum_length_map_stack)
        momentum_length_min = np.min(momentum_length_map_stack)
    else:
        momentum_length_max = radial_range[1]
        momentum_length_min = radial_range[0]

    # Because we want 300 intervals, we have to find 301 points on the line.
    ends_pre = np.linspace(momentum_length_min, momentum_length_max, num=number_of_interval + 1)

    ends = np.zeros((number_of_interval, 2))
    for l in range(number_of_interval):
        ends[l, 0] = ends_pre[l]
        ends[l, 1] = ends_pre[l + 1]

        # Get the category
    category_map = get_pixel_map(momentum_length_map_stack, ends, output_mode="in situ")

    # Get polarization correction
    polarization_correction = np.sum(np.square(np.cross(direction, np.array([1, 0, 0]))), axis=-1)

    # Get solid angle correction
    angle_correction = np.abs(np.dot(direction, np.array([0, 0, 1])))
    distance_correction = np.square(distance_detector / distance)
    geometry_correction = np.multiply(angle_correction, distance_correction)

    return category_map, momentum_length_map_stack, np.mean(ends, axis=-1), polarization_correction, geometry_correction


####################################################################
#  category map
####################################################################
def get_pixel_map(values, ends, output_mode="per class"):
    """

    Input:

    values : numpy array, values that are used to classify the indexes. 

    ends :  (M,2)-shaped numpy array. Contain the end points of each category.
            There will be M categories. At present, the interval is left open,
            and right close.

    "output_mode": String. When output_mode=="per class", the output will be of
            such shape (M, shape of "values"). When output_mode=="in situ", the
            output will be of the shape of the variable "values". Each site in 
            the output numpy array will carry a value in [0,1,2,...,M-1,M]. This 
            indicates of the specific site. Notice that there are M+1 values rather
            than M values. This is because that it is possible to have sites that
            are not in any classes. They are assigned the value M.

    Output:

    A numpy array of the shape of the variable "values" or of the shape 
    (M, the shape of "values") depending on the value of the variable "output_mode".

    """

    # Get the structure information of the input variable
    _values_shape = values.shape
    _category_number = ends.shape[0]

    if output_mode == "per class":
        # Create the mapping variable
        _class_per_site = np.zeros((_category_number + 1,) + _values_shape, dtype=bool)
        # Create a holer for simplicity
        _holder = np.zeros_like(values, dtype=bool)

        for l in range(_category_number):
            # Assign values to the mapping
            _holder[(values > ends[l, 0]) & (values <= ends[l, 1])] = True
            _class_per_site[l, :] = np.copy(_holder)

        # Get the value for the last class
        """
        Because the summation of all the boolean along the first dimension should be one. 
        The value of the last class is one minus the value of the summation of all the value
        along the first dimension

        Because the variable is initialized as zero. We can also do the summation including
        The last category.
        """

        _class_per_site[_category_number] = np.logical_not(np.sum(_class_per_site, axis=0))

        return _class_per_site

    elif output_mode == "in situ":
        # Create the mapping variable.
        _class_in_situ = np.ones_like(values, dtype=np.int32) * _category_number

        for l in range(_category_number):
            _class_in_situ[(values > ends[l, 0]) & (values <= ends[l, 1])] = l

        return _class_in_situ

    else:
        raise Exception("The value of the output_mode is invalid. Please use either \'in situ\' or \'per class\'. ")


####################################################################
#  momentum map
####################################################################

def get_momentum_map(coor_xyz, photon_energy):
    """
    Get the momentum vector for each pixel 

    :param: coor_xyz: The output of det.coor_xyz. This has to be a list.
                      The first entry of the list is the x coordinate for
                      each pixel. The second entry is the y coordinate.
                      The third entry is the distance which is also the z coordinate.
    :param: photon_energy: The photon energy in eV
    :return: A momentum vector map.
    """

    # Calculate the direction for each pixel

    # Prepare a holder
    tmp_shape = list(coor_xyz[0].shape)
    tmp_shape.append(3)
    coordinate = np.zeros(tuple(tmp_shape))

    # Assign values to this holder.
    # Because I only need to do this once, there is not need to strive for more efficiency
    coordinate[:, :, :, 0] = coor_xyz[0]
    coordinate[:, :, :, 1] = coor_xyz[1]
    coordinate[:, :, :, 2] = coor_xyz[2]

    # Get direction of pixel
    length = np.sqrt(np.sum(np.square(coordinate), axis=-1))

    direction = np.zeros_like(coordinate)
    for l in range(3):
        direction[:, :, :, l] = coordinate[:, :, :, l] / length

    # Get refracted wave
    """
    Previously, the direction is the diffraction direction. The difference between this and the forward direction
    is the scattered direction. The scattered momentum is the this difference times 2pi/ wavelength.
    """
    refracted = np.copy(direction)
    refracted[:, :, :, 2] -= 1

    # Get diffraction momentum
    wavelength = get_wavelength(photon_energy)

    refracted *= 2. * np.pi / wavelength

    return refracted, coordinate, direction, length


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
