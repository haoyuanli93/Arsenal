import sys
import numpy as np
import h5py as h5
import time

sys.path.append('/reg/neh/home5/haoyuan/Documents/my_repos/Arsenal')
import arsenal
from arsenal import PsanaUtil

runnum_list = [182, 183, 184, 185]
# Selected label value
selected_value = 3

# Project address
project_address = '/reg/d/psdm/amo/amo86615/scratch/haoyuan/waterfinder/'

# Create a holder to contain all the patterns.
total_pattern_holder = []

# Find and retrieve all the selected patterns
with h5.File(project_address + "output/water.h5", 'w') as waterfile:

    # Loop through all run numbers
    for runnum in runnum_list:

        tic = time.time()

        # Load the psocake file to get the global index
        psocake_name = str("/reg/d/psdm/amo/amo86615/scratch/" +
                           "haoyuan/psocake/r0{}/amo86615_0{}.cxi".format(runnum, runnum))

        with h5.File(psocake_name, 'r') as psocake_file:
            index_total = np.array(psocake_file['/LCLS/eventNumber'])

        # Load the label
        label_total = np.load(str(project_address + 'output/run{}/label.npy'.format(runnum)))

        # Find out the label and index
        selected_index = index_total[label_total == selected_value]

        # Create a run number entry
        new_group = waterfile.create_group("{}".format(runnum))
        new_group.create_dataset("Index", data=selected_index)

        # Get pattern number
        pattern_num = selected_index.shape[0]

        # Initialize the experiment
        det, run, times, evt, info_dict = PsanaUtil.setup_exp(exp_name='amo86615',
                                                              run_num=runnum,
                                                              det_name='pnccdBack')

        # Create a temporary holder for the patterns
        temp_holder = np.zeros((pattern_num,) + info_dict['2D pattern shape'], dtype=np.int64)

        # Loop through all the indexes:
        for local_idx in range(pattern_num):
            temp_holder[local_idx] = PsanaUtil.get_photon_2d_fast(detector=det,
                                                                  exp_run=run,
                                                                  exp_times=times,
                                                                  event_id=selected_index,
                                                                  adu_per_photon=130)

        # Save the result to the new group
        new_group.create_dataset("patterns", data=temp_holder)
        total_pattern_holder.append(np.copy(temp_holder))

        toc = time.time()

        print("It takes {:.2f} seconds to finishes run {}.".format(toc - tic, runnum))
        print("There are totally {} patterns in this run.".format(pattern_num))

    waterfile.create_dataset('all_patterns', data=np.vstack(total_pattern_holder))
