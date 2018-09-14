import sys
import os
import numpy as np
import h5py as h5
import time
import psana
import skimage.measure as skm

sys.path.append('/reg/neh/home5/haoyuan/Documents/my_repos/Arsenal')
import arsenal
import arsenal.lcls

####################################################################################################
# [USER] Specify the parameters to use
####################################################################################################
# The experiment info
exp_line = 'amo'
exp_name = 'amox26916'
user_name = 'haoyuan'
process_stage = 'scratch'
run_num = 85
det_name = 'pnccdFront'

# Specify the downsampling ratio
bin_size = 4

# Pattern format: This value should be pattern_2d or pattern_stack
pattern_format = 'pattern_2d'

# Construct the output address
output_address = '/reg/d/psdm/{}/{}/{}/{}/experiment_data/'.format(exp_line,exp_name,process_stage,user_name)
print("The output address is {}".format(output_address))

# Construct the output file name 
output_name = '{}_run_{}_all_pattern_bin_size_{}.h5'.format(exp_name, run_num, bin_size)


####################################################################################################
# [AUTO] Check the parameters
####################################################################################################
if not os.path.isdir(output_address):
    os.mkdir(output_address)
    
####################################################################################################
# [AUTO] Intialize the detector
####################################################################################################
# Get data source
ds = psana.DataSource('exp={}:run={}:idx'.format(exp_name, run_num))
run = ds.runs().next()    
env = ds.env()
times = run.times()
evt = run.event(times[0])

# Get pattern number
pattern_num = len(times)
print("There are totally {} patterns to process.".format(pattern_num))
index_to_process = np.arange(pattern_num, dtype=np.int64)

# Get detector
det = psana.Detector('pnccdFront', env)

# Get a sample
if pattern_format == 'pattern_2d':
    test_sample = arsenal.lcls.get_pattern_2d(detector=det, exp_run=run, event_id=0)
elif pattern_format == 'pattern_stack':
    test_sample = arsenal.lcls.get_pattern_stack(detector=det, exp_run=run, event_id=0)
else:
    raise Exception("pattern_format has to be either pattern_2d or pattern_stack")
    
# Get to know the sample shape
sample_shape = test_sample.shape

####################################################################################################
# [AUTO] Divide the index list
####################################################################################################
if pattern_num <= 100 :
    sub_lists = [np.arange(pattern_num, dtype=np.int64), ]
else:
    sub_lists_num = pattern_num // 100
    print("There are roughly {} batches to process.".format(sub_lists_num))
    # Get sublists 
    sub_lists = np.array_split(ary=index_to_process,indices_or_sections=sub_lists_num, axis=0)

####################################################################################################
# [AUTO] Load and downsample all the patterns
####################################################################################################
if pattern_format == 'pattern_2d':
    
    with h5.File(output_address+output_name,'w') as h5file:
        
        # For different batches
        batch_counter = 0
        
        # Loop through this list of sublists
        for sublist in sub_lists:
            
            tic = time.time()
            # First, get to know the index number in this list
            idx_num = sublist.shape[0]
            # Construct the shape of the holder variable
            holder_shape = (idx_num, ) + sample_shape
            holder = np.zeros(holder_shape)
            
            # Extract all the patterns from this sublist
            idx_counter = 0
            for idx in sublist:
                holder[idx_counter] = arsenal.lcls.get_pattern_2d(detector=det, exp_run=run, event_id=idx)
                idx_counter += 1
            
            # save_the down sampled pattern
            h5file.create_dataset('/batch_{}/index'.format(batch_counter), data=sublist)
            h5file.create_dataset('/batch_{}/pattern'.format(batch_counter), 
                                  data=skm.block_reduce(image=holder, block_size=(1, bin_size, bin_size), func=np.sum))
            
            # Update the batch_counter
            batch_counter += 1
            
            toc = time.time()
            print("It takes {:.2f} seconds to process {} patterns.".format(toc - tic, idx_num))
            
elif pattern_format == 'pattern_stack':
                                  
    with h5.File(output_address+output_name,'w') as h5file:
        
        # For different batches
        batch_counter = 0
        
        # Loop through this list of sublists
        for sublist in sub_lists:
            
            tic = time.time()
            # First, get to know the index number in this list
            idx_num = sublist.shape[0]
            # Construct the shape of the holder variable
            holder_shape = (idx_num, ) + sample_shape
            holder = np.zeros(holder_shape)
            
            # Extract all the patterns from this sublist
            idx_counter = 0
            for idx in sublist:
                holder[idx_counter] = arsenal.lcls.get_pattern_2d(detector=det, exp_run=run, event_id=idx)
                idx_counter += 1
            
            # save_the down sampled pattern
            h5file.create_dataset('/batch_{}/index'.format(batch_counter), data=sublist)
            h5file.create_dataset('/batch_{}/pattern'.format(batch_counter), 
                                  data=skm.block_reduce(image=holder, block_size=(1, bin_size, bin_size), func=np.sum))
            
            # Update the batch_counter
            batch_counter += 1
            
            toc = time.time()
            print("It takes {:.2f} seconds to process {} patterns.".format(toc - tic, idx_num))
else:
    raise Exception("pattern_format has to be either pattern_2d or pattern_stack")
