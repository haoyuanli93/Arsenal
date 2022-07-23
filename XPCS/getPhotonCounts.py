"""
This script tries to get the photon counts
per ePix detector and per q map with MPI
parallel calculation.
"""

import time

import numpy as np
from mpi4py import MPI

# Initialize the MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()
batch_num = comm_size

"""
Preparation step: Define the function
"""


def reconstruct_img_4ePix(photons_x, photons_y, shape):
    nx, ny = shape
    phot_img1, _, _ = np.histogram2d(photons_y[0] + 0.5, photons_x[0] + 0.5,
                                     bins=[np.arange(nx + 1), np.arange(ny + 1)])
    phot_img2, _, _ = np.histogram2d(photons_y[1] + 0.5, photons_x[1] + 0.5,
                                     bins=[np.arange(nx + 1), np.arange(ny + 1)])
    phot_img3, _, _ = np.histogram2d(photons_y[2] + 0.5, photons_x[2] + 0.5,
                                     bins=[np.arange(nx + 1), np.arange(ny + 1)])
    phot_img4, _, _ = np.histogram2d(photons_y[3] + 0.5, photons_x[3] + 0.5,
                                     bins=[np.arange(nx + 1), np.arange(ny + 1)])

    return np.stack([phot_img1, phot_img2, phot_img3, phot_img4])


def get_probability_per_epix_and_per_q_map(photon_x_list, photon_y_list, q_category_holder):
    # Get the pattern
    pattern_num = photon_x_list.shape[1]
    print("There are totally {:.2e} patterns to analyze".format(pattern_num))

    # Get the photon count holder
    photon_count_holder = np.zeros((pattern_num, 4, 5, 5), dtype=np.float64)

    # Loop through each pattern and get the corresponding 2d image
    pattern_holder = np.zeros((4, 704, 768))
    tic = time.time()
    for pattern_idx in range(pattern_num):
        pattern_holder[:, :, :] = reconstruct_img_4ePix(photons_x=photon_x_list[:, pattern_idx],
                                                        photons_y=photon_y_list[:, pattern_idx],
                                                        shape=(704, 768))

        # Get the photon counts
        # Loop through each epix
        for epix_idx in range(4):
            # Loop though q bins
            for q_idx in range(5):
                hist, bins = np.histogram(
                    pattern_holder[epix_idx][q_category_holder[epix_idx] == q_idx + 1],
                    bins=5, range=(0.5, 5.5))
                photon_count_holder[pattern_idx, epix_idx, q_idx, :] = hist[:]

        # If processed 1000 patterns: print time
        if ((pattern_idx + 1) % 1000) == 0:
            toc = time.time()
            print("It takes {:.2f} seconds to process 1000 patterns".format(toc - tic))
            tic = time.time()

    return photon_count_holder


"""
Step one: All the node load all the information
"""
# Load the photon count info
photon_x_cc = np.load("./data/photon_x_cc_run_174_187.npy", allow_pickle=True)
photon_x_vcc = np.load("./data/photon_x_vcc_run_174_187.npy", allow_pickle=True)
photon_x_both = np.load("./data/photon_x_both_run_174_187.npy", allow_pickle=True)

photon_y_cc = np.load("./data/photon_y_cc_run_174_187.npy", allow_pickle=True)
photon_y_vcc = np.load("./data/photon_y_vcc_run_174_187.npy", allow_pickle=True)
photon_y_both = np.load("./data/photon_y_both_run_174_187.npy", allow_pickle=True)

# Load the q bin map
category_map = np.load("./data/category_mask_per_epix.npy")

# Get all the masks
mask_holder = np.zeros_like(category_map, dtype=bool)
for x in range(4):
    mask_holder[x, :, :] = np.load(
        "/reg/d/psdm/xpp/xppc00120/results/haoyuan/Analysis_v1/epix_{}_mask.npy".format(x + 1)).astype(bool)

category_map[np.logical_not(mask_holder)] = 0

"""
Step Two: Each node calculate its job
"""
# CC beam
indexes = np.linspace(start=0, stop=photon_x_cc.shape[1], num=batch_num + 1)
indexes = indexes.astype(np.int64)

photon_counts_cc = get_probability_per_epix_and_per_q_map(
    photon_x_list=photon_x_cc[:, indexes[comm_rank]:indexes[comm_rank + 1]],
    photon_y_list=photon_y_cc[:, indexes[comm_rank]:indexes[comm_rank + 1]],
    q_category_holder=category_map)

# VCC beam
indexes = np.linspace(start=0, stop=photon_x_vcc.shape[1], num=batch_num + 1)
indexes = indexes.astype(np.int64)

photon_counts_vcc = get_probability_per_epix_and_per_q_map(
    photon_x_list=photon_x_vcc[:, indexes[comm_rank]:indexes[comm_rank + 1]],
    photon_y_list=photon_y_vcc[:, indexes[comm_rank]:indexes[comm_rank + 1]],
    q_category_holder=category_map)

# Both beam
indexes = np.linspace(start=0, stop=photon_x_both.shape[1], num=batch_num + 1)
indexes = indexes.astype(np.int64)

photon_counts_both = get_probability_per_epix_and_per_q_map(
    photon_x_list=photon_x_both[:, indexes[comm_rank]:indexes[comm_rank + 1]],
    photon_y_list=photon_y_both[:, indexes[comm_rank]:indexes[comm_rank + 1]],
    q_category_holder=category_map)

comm.Barrier()  # Synchronize
photon_counts_cc_collect = comm.gather(photon_counts_cc, root=0)
photon_counts_vcc_collect = comm.gather(photon_counts_vcc, root=0)
photon_counts_both_collect = comm.gather(photon_counts_both, root=0)
"""
Step Three: The master node collect and assemble the results
"""
if comm_rank == 0:
    photon_counts_cc_tot = np.concatenate(photon_counts_cc_collect, axis=0)
    photon_counts_vcc_tot = np.concatenate(photon_counts_vcc_collect, axis=0)
    photon_counts_both_tot = np.concatenate(photon_counts_both_collect, axis=0)

    np.save("./data/photon_count_cc.npy", photon_counts_cc_tot)
    np.save("./data/photon_count_vcc.npy", photon_counts_vcc_tot)
    np.save("./data/photon_count_both.npy", photon_counts_both_tot)
