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