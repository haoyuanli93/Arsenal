import numpy as np
import h5py as h5
import arsenal.util

"""
This module focuses on applications in LCLS that  does not require psana
"""


###########################################################################################################
# Psocake
###########################################################################################################
def cast_numpy_to_txt(arr, output_file):
    """
    In psocake, in spi mode, current, the generate average pattern function only produce
    txt files for max. However, the mean patterns are also desirable for distance calibration.
    Therefore, this function can be used to cast the stack pattern, numpy array to a text file.

    :param arr: The numpy array to cast.
    :param output_file: The output file that is compatible with calibman
    :return: None
    """
    shape = arr.shape
    arr = arr.reshape([shape[0] * shape[1], shape[2]])

    np.savetxt(fname=output_file, X=arr, delimiter=' ', fmt='%.18e', newline='\n', )


def cast_txt_to_numpy(iuput_file):
    """
    The inverse function of the previous function.
    This one read the txt file and cast that into a numpy array

    :param iuput_file: The txt file to process.
    :return: The array.
    """
    # Load the txt file
    with open(iuput_file, 'r') as tmpfile:
        lines = tmpfile.readlines()

        # Restore the numpy array
        holder = []
        for line in lines:
            holder.append([float(x) for x in line.split(' ')])

        # Construct the numpy array
        holder = np.array(holder)

    return holder


def get_cxi_file_position(exp_line, exp_name, user_name, process_stage, run_num):
    """
    Get the cxi file position saved by the psocake

    :param exp_line: The experiment line: AMO or CXI or ...
    :param exp_name: The experiment name: amo86615 amox26916 or ...
    :param user_name: The user name
    :param process_stage: The process stage: scratch or results ...
    :param run_num: The run number
    :return:
    """

    # Construct the file address of the corresponding cxi file
    file_name = '/reg/d/psdm/{}/{}/{}/{}/psocake/r{:0>4d}/{}_{:0>4d}.cxi'.format(exp_line,
                                                                                 exp_name,
                                                                                 process_stage,
                                                                                 user_name,
                                                                                 run_num,
                                                                                 exp_name,
                                                                                 run_num)
    return file_name


def get_cxi_photon_energy(exp_line, exp_name, user_name, process_stage, run_num):
    """
    Get the photon energy from the cxi file

    :param exp_line: The experiment line: AMO or CXI or ...
    :param exp_name: The experiment name: amo86615 amox26916 or ...
    :param user_name: The user name
    :param process_stage: The process stage: scratch or results ...
    :param run_num: The run number
    :return:
    """
    # Construct the file address of the corresponding cxi file
    file_name = get_cxi_file_position(exp_line=exp_line,
                                      exp_name=exp_name,
                                      process_stage=process_stage,
                                      user_name=user_name,
                                      run_num=run_num)
    # Get photon energy
    with h5.File(file_name, 'r') as h5file:
        holder = h5file['/LCLS/photon_wavelength_A'].value
        # convert to meter
        photon_wavelength = holder[0] / (10 ** 10)
        photon_energy = arsenal.util.get_energy(wavelength=photon_wavelength)

    return photon_energy


def get_cxi_pattern_idx(exp_line, exp_name, user_name, process_stage, run_num):
    """
    Get the photon energy from the cxi file

    :param exp_line: The experiment line: AMO or CXI or ...
    :param exp_name: The experiment name: amo86615 amox26916 or ...
    :param user_name: The user name
    :param process_stage: The process stage: scratch or results ...
    :param run_num: The run number
    :return:
    """
    # Construct the file address of the corresponding cxi file
    file_name = get_cxi_file_position(exp_line=exp_line,
                                      exp_name=exp_name,
                                      process_stage=process_stage,
                                      user_name=user_name,
                                      run_num=run_num)
    # Get photon energy
    with h5.File(file_name, 'r') as h5file:
        holder = h5file['/LCLS/photon_wavelength_A'].value
        # convert to meter
        photon_wavelength = holder[0] / (10 ** 10)
        photon_energy = arsenal.util.get_energy(wavelength=photon_wavelength)

    return photon_energy
