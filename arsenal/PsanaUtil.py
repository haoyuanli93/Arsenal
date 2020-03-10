import numpy as np
import psana

import arsenal.geometry as ag
import arsenal.physics as ap

"""
This module focuses on applications in LCLS that requires psana
"""


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
    :return: detector object, run object, times object, 2d pattern shape, pattern stack shape,
            mask stack, 2d mask
    """

    # Initialize the datasource
    ds = psana.DataSource('exp={}:run={}:idx'.format(exp_name, run_num))
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

    return det, run, times, evt, holder


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


def get_photon_stack(detector, exp_run, event_id, adu_per_photon):
    """
    Get a photon stack from the detector from the specific run for the specified event id

    :param detector: The detector object
    :param exp_run: The run object
    :param event_id: The event it
    :param adu_per_photon:
    :return: The pattern stack
    """
    times = exp_run.times()
    evt = exp_run.event(times[event_id])
    pattern_stack = detector.photons(evt=evt, adu_per_photon=adu_per_photon)
    return pattern_stack


def get_photon_2d(detector, exp_run, event_id, adu_per_photon):
    """
    Get a photon number pattern from the detector from the specific run for the specified event id

    :param detector: The detector object
    :param exp_run: The run object
    :param event_id: The event it
    :param adu_per_photon:
    :return: The pattern
    """
    times = exp_run.times()
    evt = exp_run.event(times[event_id])
    pattern_stack = detector.photons(evt=evt, adu_per_photon=adu_per_photon)
    pattern_2d = detector.image(evt=evt, nda_in=pattern_stack)
    return pattern_2d


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


def get_photon_stack_fast(detector, exp_run, exp_times, event_id, adu_per_photon):
    """
    Get a photon stack from the detector from the specific run for the specified event id

    :param detector: The detector object
    :param exp_run: The run object
    :param exp_times: The time object
    :param event_id: The event it
    :param adu_per_photon:
    :return: The pattern stack
    """
    evt = exp_run.event(exp_times[event_id])
    pattern_stack = detector.photons(evt=evt, adu_per_photon=adu_per_photon)
    return pattern_stack


def get_photon_2d_fast(detector, exp_run, exp_times, event_id, adu_per_photon):
    """
    Get a photon number pattern from the detector from the specific run for the specified event id

    :param detector: The detector object
    :param exp_run: The run object
    :param exp_times: The time object
    :param event_id: The event it
    :param adu_per_photon:
    :return: The pattern
    """
    evt = exp_run.event(exp_times[event_id])
    pattern_stack = detector.photons(evt=evt, adu_per_photon=adu_per_photon)
    pattern_2d = detector.image(evt=evt, nda_in=pattern_stack)
    return pattern_2d


def get_pixel_info(detector, run_num):
    """
    Get the pixel position of the detector in the unit of meter.

    :param detector: The pnccd detector object
    :param run_num:
    :return: A dictionary containing useful info.
    """

    # Get detector pixel information
    tmp_pos = detector.coords_xyz(par=run_num)
    pixel_areas_stack = detector.areas(par=run_num)
    pixel_areas_stack *= 1e-12  # convert the unit to meter

    # Get shape and pixel number
    stack_shape = detector.pedestals(par=run_num).shape
    pixel_num = int(np.prod(stack_shape))
    pixel_areas_1d = np.reshape(pixel_areas_stack, (pixel_num,))

    # Extrach pixel position info
    pixel_position_1d = np.zeros((pixel_num, 3), dtype=np.float64)
    for l in range(3):
        pixel_position_1d[:, l] = np.reshape(tmp_pos[l], pixel_num)

    # Convert to meter
    pixel_position_1d *= 1e-6

    # Get the pixel distance and direction
    pixel_distances_1d = np.sqrt(np.sum(np.square(pixel_position_1d), axis=-1))
    pixel_directions_1d = np.divide(pixel_position_1d, pixel_distances_1d[:, np.newaxis])

    # Get the stack shape
    pixel_position_stack = np.reshape(pixel_position_1d, stack_shape + (3,))
    pixel_directions_stack = np.reshape(pixel_directions_1d, stack_shape + (3,))
    pixel_distance_stack = np.reshape(pixel_distances_1d, stack_shape)

    # Assemble the results in to dictionary
    info = {'pixel num': pixel_num,
            'stack shape': stack_shape,

            'pixel position 1d': pixel_position_1d,
            'pixel distance 1d': pixel_distances_1d,
            'pixel area 1d': pixel_areas_1d,
            'pixel direction 1d': pixel_directions_1d,

            'pixel position stack': pixel_position_stack,
            'pixel distance stack': pixel_distance_stack,
            'pixel area stack': pixel_areas_stack,
            'pixel direction stack': pixel_directions_stack}

    return info


def get_detector_polarization_correction(detector, run_num, polarization):
    """
    Get the detector polarization correction based on the detector object in psana.

    :param detector: psana detector object.
    :param polarization: The polarization of the beam.
    :param run_num: The run number of the detector object
    :return:
    """
    pixel_info = get_pixel_info(detector=detector, run_num=run_num)

    return ap.get_polarization_correction(pixel_position_m=pixel_info['pixel position 1d'],
                                          reference_position_m=np.zeros(3, dtype=np.float64),
                                          polarization=polarization)


def get_detector_solid_angle_per_pixel(detector, run_num):
    """
    Get the detector polarization correction based on the detector object in psana.

    :param detector: psana detector object.
    :param run_num: The run number of the detector object
    :return:
    """
    pixel_info = get_pixel_info(detector=detector, run_num=run_num)

    pixel_orientation = np.zeros((pixel_info['pixel num'], 3), dtype=np.float64)
    pixel_orientation[:, 2] = 1

    solid_angle = ag.get_solid_angle(pixel_area_m=pixel_info['pixel area 1d'],
                                     pixel_position_m=pixel_info['pixel position 1d'],
                                     pixel_orientation=pixel_orientation,
                                     reference_point_m=np.zeros(3, dtype=np.float64))

    return solid_angle


