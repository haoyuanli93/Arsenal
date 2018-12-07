import numpy as np
import h5py as h5

# Get the pattern number for each file
patterns_num = 1000

# Construct the run nummber list and the file name list
runnum_list = [186, 187, 190, 191, 192, 193, 194, 196, 197]


# Define a function to get the file name
def get_input_file_name(run_num):
    """
    This is just a quick function to get the file name.
    :param run_num:
    :return:
    """
    return str("/reg/d/psdm/amo/amo86615/scratch" +
               "/haoyuan/experiment_data/" +
               "amo86615_run_{}_selection_based_on_psocake.h5".format(run_num))


with h5.File("/reg/d/psdm/amo/amo86615/scratch/"
             "haoyuan/experiment_data/amo86615_sample.h5", 'w') as outputfile:
    # Loop through the run number list
    for runnum in runnum_list:

        print(runnum)

        # get file name
        input_file_name = get_input_file_name(run_num=runnum)

        # Create a holder
        holder = np.zeros((patterns_num, 260, 257))
        # Load the h5 file
        with h5.File(input_file_name) as h5file:

            # Get dataset number
            dataset_num = int(len(list(h5file.keys())))
            dataset_choice = np.random.randint(low=0, high=dataset_num - 1, size=patterns_num)

            for idx in range(patterns_num):

                # Dataset name
                dataset_name = "batch_{}_pattern".format(dataset_choice[idx])

                dataset = h5file[dataset_name]
                dataset_pattern_num = dataset.shape[0]
                local_idx = np.random.randint(low=0, high=dataset_pattern_num-1, size=1)[0]

                # Load the pattern
                holder[idx] = np.array(dataset[local_idx])

        outputfile.create_dataset("{}".format(runnum), data=holder)
