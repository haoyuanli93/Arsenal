import psana
import numpy as np


###########################################################################################################
# Data IO
###########################################################################################################
def setup_exp(exp_name, run_num, det_name,
              mask_calib_on=True,
              mask_status_on=True,
              mask_edges_on=True,
              mask_central_on=True,
              mask_unbond_on=True,
              mask_unbondnrs_on=True):
    """
    This function is used to initialize the detector object. The return is a dictionary.
    The name of the dictionary is suggestive.

    :param exp_name: The experiment name
    :param run_num: The run number to inspect
    :param det_name: The detector name
    :param mask_calib_on: 
    :param mask_status_on: 
    :param mask_edges_on: 
    :param mask_central_on: 
    :param mask_unbond_on: 
    :param mask_unbondnrs_on: 
    :return: detector object, run object, times object, 2d pattern shape, pattern stack shape, mask stack, 2d mask
    """

    # Initialize the datasource
    ds = psana.DataSource('exp={}:run{}:idx'.format(exp_name, run_num))
    run = ds.runs().next()
    times = run.times()
    env = ds.env()
    evt = run.event(times[0])

    # Get the detector
    det = psana.Detector(det_name, env)

    # Generate a holder for some experiment information
    holder = {}

    # Get pattern shapes
    example = det.image(evt)
    holder.update({'Example 2D': example,
                   '2D pattern shape': example.shape})

    # Get pattern stack shape
    example = det.calib(evt)
    holder.update({'Example stack': example,
                   'Pattern stack shape': example.shape})

    # get mask and save mask
    mask_stack = det.mask(evt, calib=mask_calib_on, status=mask_status_on,
                          edges=mask_edges_on, central=mask_central_on,
                          unbond=mask_unbond_on, unbondnbrs=mask_unbondnrs_on)
    mask_2d = det.image(evt, mask_stack)
    holder.update({'Mask 2D': mask_2d,
                   'Mask stack': mask_stack})

    return det, run, times, holder


# Get a calibrated sample
def get_pattern_stack(detector, exp_run, event_id):
    """
    Get a pattern stack from the detector from the specific run for the specified event id

    :param detector: The detector object
    :param exp_run: The run object
    :param event_id: The event it
    :return: The pattern stack
    """
    times = exp_run.times()
    evt = exp_run.event(times[event_id])
    pattern_stack = detector.calib(evt)
    return pattern_stack


def get_pattern_2d(detector, exp_run, event_id):
    """
    Get a pattern from the detector from the specific run for the specified event id

    :param detector: The detector object
    :param exp_run: The run object
    :param event_id: The event it
    :return: The pattern
    """
    times = exp_run.times()
    evt = exp_run.event(times[event_id])
    pattern_2d = detector.image(evt)
    return pattern_2d


# Get a photon number sample
def get_photon_stack(detector, exp_run, event_id):
    """
    Get a photon stack from the detector from the specific run for the specified event id

    :param detector: The detector object
    :param exp_run: The run object
    :param event_id: The event it
    :return: The pattern stack
    """
    times = exp_run.times()
    evt = exp_run.event(times[event_id])
    pattern_stack = detector.photons(evt)
    return pattern_stack


def get_photon_2d(detector, exp_run, event_id):
    """
    Get a photon number pattern from the detector from the specific run for the specified event id

    :param detector: The detector object
    :param exp_run: The run object
    :param event_id: The event it
    :return: The pattern
    """
    times = exp_run.times()
    evt = exp_run.event(times[event_id])
    pattern_stack = detector.photons(evt=evt)
    pattern_2d = detector.image(evt=evt, nda_in=pattern_stack)
    return pattern_2d


# Get a calibrated sample
def get_pattern_stack_fast(detector, exp_run, exp_times, event_id):
    """
    Get a pattern stack from the detector from the specific run for the specified event id

    :param detector: The detector object
    :param exp_run: The run object
    :param exp_times: The time object
    :param event_id: The event it
    :return: The pattern stack
    """
    evt = exp_run.event(exp_times[event_id])
    pattern_stack = detector.calib(evt)
    return pattern_stack


def get_pattern_2d_fast(detector, exp_run, exp_times, event_id):
    """
    Get a pattern from the detector from the specific run for the specified event id

    :param detector: The detector object
    :param exp_run: The run object
    :param exp_times: The time object
    :param event_id: The event it
    :return: The pattern
    """
    evt = exp_run.event(exp_times[event_id])
    pattern_2d = detector.image(evt)
    return pattern_2d


# Get a photon number sample
def get_photon_stack_fast(detector, exp_run, exp_times, event_id):
    """
    Get a photon stack from the detector from the specific run for the specified event id

    :param detector: The detector object
    :param exp_run: The run object
    :param exp_times: The time object
    :param event_id: The event it
    :return: The pattern stack
    """
    evt = exp_run.event(exp_times[event_id])
    pattern_stack = detector.photons(evt)
    return pattern_stack


def get_photon_2d_fast(detector, exp_run, exp_times, event_id):
    """
    Get a photon number pattern from the detector from the specific run for the specified event id

    :param detector: The detector object
    :param exp_run: The run object
    :param exp_times: The time object
    :param event_id: The event it
    :return: The pattern
    """
    evt = exp_run.event(exp_times[event_id])
    pattern_stack = detector.photons(evt=evt)
    pattern_2d = detector.image(evt=evt, nda_in=pattern_stack)
    return pattern_2d


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
