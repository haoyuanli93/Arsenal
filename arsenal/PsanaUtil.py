import psana

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
    :return: detector object, run object, times object, 2d pattern shape, pattern stack shape, mask stack, 2d mask
    """

    # Initialize the datasource
    ds = psana.datasource('exp={}:run={}:idx'.format(exp_name, run_num))
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
