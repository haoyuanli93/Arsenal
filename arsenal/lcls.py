import psana


###########################################################################################################
# Data IO
###########################################################################################################


def setup_exp(exp_name, run_num,
              det_name,
              mask_calib_on=True,
              mask_status_on=True,
              mask_edges_on=True,
              mask_central_on=True,
              mask_unbond_on=True,
              mask_unbondnrs_on=True):
    """
    
    :param exp_name: 
    :param run_num: 
    :param mask_calib_on: 
    :param mask_status_on: 
    :param mask_edges_on: 
    :param mask_central_on: 
    :param mask_unbond_on: 
    :param mask_unbondnrs_on: 
    :return: 
    """

    ds = psana.DataSource('exp={}:run{}:idx'.format(exp_name, run_num))
    run = ds.runs().next()
    times = run.times()
    env = ds.env()

    det = psana.Detector(det_name, env)
    et = psana.EventTime(int(ts[0]), fid[0])
    evt = run.event(et)
    example = det.image(evt)
    shape = example.shape

    # get mask and save mask
    mask_stack = det.mask(evt, calib=mask_calib_on, status=mask_status_on,
                          edges=mask_edges_on, central=mask_central_on,
                          unbond=mask_unbond_on, unbondnbrs=mask_unbondnrs_on)
    mask_2d = det.image(evt, mask_stack)

    return


# Get a sample
def get_pattern_stack(detector, exp_run, event_id):
    times = exp_run.times()
    evt = exp_run.event(times[event_id])
    pattern_stack = detector.calib(evt)
    return pattern_stack


def get_pattern_2d(detector, exp_run, event_id):
    times = exp_run.times()
    evt = exp_run.event(times[event_id])
    pattern_2d = detector.image(evt)
    return pattern_2d
