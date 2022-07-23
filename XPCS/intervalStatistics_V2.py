import numpy as np
import h5py as h5
import time
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Parameters')
# Optional argument
parser.add_argument('--runNum', type=int,
                    help='Run num to process')
args = parser.parse_args()

"""
The general strategy of my analysis is to 
1. Count the actual statistics
2. Calculate the theoretical statistics
3. Compare both of them.

In this script, I only want to solve the first problem
1. Loop through all the patterns
    a. Count the number of events, that a single pixel lights up continuously for 2, 3, 4, 5, 6 patterns, for every 120 patterns
        i. Divide the whole dataset into subsets with pattern number of 120
        ii. Cumulatively get the statistics.
        iii. Save the statistics for each of these sub-dataset 
        vi. Also save a data set about the statistics of the total run. 
    b. Record the condition of CC and VCC things to make the comparison more consistent
    c. Maybe there are some issue with the common mode corrections. If this is the case, it should reveal itself through our statistics
"""


########################################################################################
#     Define functions
########################################################################################
# Reconstruct the image from the photon distribution
def reconstruct_img(photons_x, photons_y, shape):
    nx, ny = shape
    phot_img, _, _ = np.histogram2d(photons_y + 0.5, photons_x + 0.5, bins=[np.arange(nx + 1), np.arange(ny + 1)])
    return phot_img


########################################################################################
#     Load the data
########################################################################################
run_num = int(args.runNum)

# Load a run
with h5.File("/reg/d/psdm/xpp/xppc00120/results/smalldata_output_hy/" +
             "xppc00120_Run0{}.h5".format(run_num)) as datafile:
    # Load the intensity level indicator
    diode2 = np.array(datafile['diode2/channels'])
    diodeU = np.array(datafile['diodeU/channels'])
    ipm2 = np.array(datafile['ipm2/sum'])
    l3e = np.array(datafile['ebeam/L3_energy'])

    # Load the cc/vcc branches indicator
    ccDataHolder = np.array(datafile['ai/ch06'])
    vccDataHolder = np.array(datafile['ai/ch07'])

    photonX = []
    photonY = []
    # Loop though the epix detectors to get the photons
    for epix_idx in range(1, 5):
        photonX.append(np.array(datafile['epix_alc{}/ragged_droplet_photon_i'.format(epix_idx)]))
        photonY.append(np.array(datafile['epix_alc{}/ragged_droplet_photon_j'.format(epix_idx)]))

########################################################################################
#     Divide the data
########################################################################################
# Get photon interval statistics
patternNum = ccDataHolder.shape[0]
batchSize = 120

# Get the indeses for the sub-batches
batchEnds = np.linspace(0, patternNum, num=int(patternNum // batchSize) + 2, ).astype(np.int64)
batchNum = batchEnds.shape[0] - 1
print("There are {} batches in run {} to process".format(batchNum, run_num))
########################################################################################
#     Get interval statistics
########################################################################################
# Save the statistics
with h5.File("/reg/d/psdm/xpp/xppc00120/results/haoyuan/Output/xppc00120_Run0{}_intervalStatistics.h5".format(run_num),
             'w') as outputFile:
    # Loop through the batches
    for batchIdx in range(batchNum):
        tic = time.time()
        # Get the number of patterns in this batch
        batchSizeLocal = batchEnds[batchIdx + 1] - batchEnds[batchIdx]
        if batchSizeLocal < 6:
            print("This batch is smaller than 6 patterns. Go to the next batch.")
            continue

        # Extract the batch of the data
        ccDataBatch = ccDataHolder[batchEnds[batchIdx]:batchEnds[batchIdx + 1]]
        vccDataBatch = vccDataHolder[batchEnds[batchIdx]:batchEnds[batchIdx + 1]]

        #  Separate CC, VCC data
        ccIdx = np.zeros_like(ccDataBatch, dtype=bool)
        ccIdx[ccDataBatch > 2] = True
        vccIdx = np.zeros_like(vccDataBatch, dtype=bool)
        vccIdx[vccDataBatch > 2] = True

        bothIdx = np.logical_and(ccIdx, vccIdx)
        ccIdx[bothIdx] = False
        vccIdx[bothIdx] = False

        # Get a interval holder
        intervalStatistics = np.zeros((5, 4, 704, 768))
        intervalStatisticsCC = np.zeros((5, 4, 704, 768))
        intervalStatisticsVCC = np.zeros((5, 4, 704, 768))
        intervalStatisticsBoth = np.zeros((5, 4, 704, 768))

        # Create time windows for the statistics
        timeWindowS = [np.zeros((windowIdx, 4, 704, 768), dtype=bool) for windowIdx in range(2, 7)]
        ccStatusWindows = [np.zeros(windowIdx, dtype=bool) for windowIdx in range(2, 7)]
        vccStatusWindows = [np.zeros(windowIdx, dtype=bool) for windowIdx in range(2, 7)]
        bothStatusWindows = [np.zeros(windowIdx, dtype=bool) for windowIdx in range(2, 7)]

        # Get the 2D pattern
        patternHolder = np.zeros((4, 704, 768), dtype=bool)
        for patternIdx in range(batchEnds[batchIdx], batchEnds[batchIdx + 1]):

            # Reconstruct the pattern per ePix
            for ePixIdx in range(4):
                patternHolder[ePixIdx, :, :] = reconstruct_img(photons_x=photonX[ePixIdx][patternIdx],
                                                               photons_y=photonY[ePixIdx][patternIdx],
                                                               shape=(704, 768)).astype(bool)

            # Append the pattern to the time window in a cyclic way, and count the events of consecutive True
            for x in range(5):
                # Get the pattern and cc, vcc status
                timeWindowS[x][patternIdx % (x + 2), :, :, :] = patternHolder[:, :, :]
                ccStatusWindows[x][patternIdx % (x + 2)] = ccIdx[patternIdx]
                vccStatusWindows[x][patternIdx % (x + 2)] = vccIdx[patternIdx]
                bothStatusWindows[x][patternIdx % (x + 2)] = bothIdx[patternIdx]

                # Calculate the statistics regardless of the branching status
                sample = np.all(timeWindowS[x], axis=0).astype(np.int64)
                intervalStatistics[x, :, :, :] += sample
                # Calculate the statistics considering the branching status
                if np.all(ccStatusWindows[x]):
                    intervalStatisticsCC[x, :, :, :] += sample
                elif np.all(vccStatusWindows[x]):
                    intervalStatisticsVCC[x, :, :, :] += sample
                elif np.all(bothStatusWindows[x]):
                    intervalStatisticsBoth[x, :, :, :] += sample
                else:
                    print("Branch switching.")

        # Save the data
        intervalStatisticsDataSet = outputFile.create_dataset("intervalStatistics_Batch_{}".format(batchIdx), data=intervalStatistics)
        intervalStatisticsDataSetCC = outputFile.create_dataset("intervalStatistics_CC_Batch_{}".format(batchIdx), data=intervalStatisticsCC)
        intervalStatisticsDataSetVCC = outputFile.create_dataset("intervalStatistics_VCC_Batch_{}".format(batchIdx), data=intervalStatisticsVCC)
        intervalStatisticsDataSetBoth = outputFile.create_dataset("intervalStatistics_Both_Batch_{}".format(batchIdx), data=intervalStatisticsBoth)

        toc = time.time()
        print("It takes {:.1f} seconds to finish batch {}".format(toc - tic, batchIdx))
