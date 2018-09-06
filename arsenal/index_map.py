import numpy


# import h5py


def get_index_map_numpy(txt_file):
    """
    This function generate a numpy array that contains the map from global indexes to local indexes.

    :param txt_file: This is the txt file that contains the numpy arrays to process
    :return: a numpy array containing the map
             [[file index, local index, global index],
              [file index, local index, global index],
                            ...
              [file index, local index, global index]]
    """

    # Read the lines. Do not change the order of the files in the txt file.
    with open(txt_file, 'r') as txtFile:
        lines = txtFile.readlines()

    # Remove redundant "/n" symbol and blank spaces
    lines = [x.strip('\n') for x in lines]
    lines = [x.strip() for x in lines]

    # Holder for data_num
    data_num_list = [0, ]  # The index is set to zero to facilitate index manipulation
    file_num = len(lines)

    # Loop through all files
    for numpy_file in lines:
        data = numpy.load(numpy_file)
        data_num = data.shape[0]
        data_num_list.append(data_num)

    # Create the map
    data_num_tot = int(numpy.sum(numpy.array(data_num_list, dtype=numpy.int)))
    index_map = numpy.zeros((data_num_tot, 3), dtype=numpy.int)

    for l in range(file_num):
        # File indexes
        index_map[data_num_list[l]:data_num_list[l + 1], 0] = l

        # Local indexes
        index_map[data_num_list[l]:data_num_list[l + 1], 1] = numpy.arange(data_num_list[l], dtype=numpy.int)

    # Global indexes
    index_map[:, 2] = numpy.arange(data_num_tot, dtype=numpy.int)

    return index_map
