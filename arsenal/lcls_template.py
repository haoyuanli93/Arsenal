import psana

###########################################################################################################
# Data IO
###########################################################################################################
exp_line = 'amo'
exp_name = 'amox26916'
user_name = 'haoyuan'

process_stage = 'scratch'

run_num = 85
det_name = 'pnccdFront'

# Construct the file address of the corresponding cxi file
file_name = '/reg/d/psdm/{}/{}/{}/{}/psocake/r{:0>4d}/{}_{:0>4d}.cxi'.format(exp_line,
                                                                             exp_name,
                                                                             process_stage,
                                                                             user_name,
                                                                             run_num,
                                                                             exp_name,
                                                                             run_num)
print("The cxi file is located at {}".format(file_name))


def setup_exp(exp_name, run_num, det_name,
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
