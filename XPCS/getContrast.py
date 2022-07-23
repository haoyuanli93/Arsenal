"""
I create this file because the original way to calculate the contrast is time consuming.
I want to make it more automatic.
"""
import numpy as np
from scipy import stats
import time


class fancyFont:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def chiSquare(p, kMean, M, nroi):
    """
    Get the chi square test value for different conditions.

    :param p:
    :param kMean:
    :param M:
    :param nroi:
    :return:
    """
    # Get the number of different kMean to fit
    kMeanNum = np.size(kMean)

    # Create holder of k to calculate the probability for different k and kbar
    k = np.zeros((kMeanNum, 4))
    k[:, 0] = 0.
    k[:, 1] = 1.
    k[:, 2] = 2.
    k[:, 3] = 3.

    # Duplicate kMean to match the size of k to calculate the probability distribution for fitting
    kMeanHolder = np.zeros((kMeanNum, 4))
    kMeanHolder[:, :] = kMean[:, np.newaxis]

    prob_distribution = stats.nbinom.pmf(k=k, n=M, p=1. / (1. + kMeanHolder / M))

    # Calculate the chi square value.
    return 2.0 * np.nansum(p * np.log(p / prob_distribution)) * nroi


def getContrastFromProbability(probability, kMean, nroi, Mmin=1, Mmax=100., Mnum=50):
    """
    Calculate the chi square value at each specified M
    Find the value where chi square result is minimal.
    The inverse of the corresponding M is the contrast.

    :param probability: Probabilities calculated from each pattern (number of pattern, 4).
                        probability[0] = (P0, P1, P2, P3)
    :param kMean: The kbar per pattern. Shape (number of pattern, )
    :param nroi: Number of pixels in the ROI
    :param Mmin:
    :param Mmax:
    :param Mnum:50
    :return:
    """

    # The list of M to calculate the chi square test.
    Ms = np.linspace(Mmin, Mmax, num=Mnum)

    # Create a holder for chi square
    chi2 = np.zeros(Ms.size)

    # Loop through all M
    for idx in range(Mnum):
        chi2[idx] = chiSquare(p=probability[:, :4],
                              kMean=kMean,
                              M=Ms[idx],
                              nroi=nroi)

    # Get the M where the chi square is minimal
    pos = np.argmin(chi2)
    M0 = Ms[pos]

    # Estimate the uncertainty of the value.
    # curvature as error analysis
    dM = Ms[1] - Ms[0]
    delta_M = np.sqrt(dM ** 2 / (chi2[pos + 1] + chi2[pos - 1] - 2 * chi2[pos]))

    return M0, delta_M, chi2


def getContrast(probability, kMean, nroi):
    """
    Previously, I manually change the searching range and resolution.
    This is boring.
    This time, I make this process automatic.
    """
    # The first iteration of calculation
    tic = time.time()
    M0, delta_M, chi2 = getContrastFromProbability(probability=probability,
                                                   kMean=kMean,
                                                   nroi=nroi,
                                                   Mmin=1,
                                                   Mmax=100,
                                                   Mnum=50)
    toc = time.time()
    print("It takes {:.2f} seconds to finish the first iteration of searching".format(toc - tic))

    # The second iteration of calculation
    tic = time.time()
    M0, delta_M, chi2 = getContrastFromProbability(probability=probability,
                                                   kMean=kMean,
                                                   nroi=nroi,
                                                   Mmin=max(1, M0 - 4),
                                                   Mmax=M0 + 4,
                                                   Mnum=50)
    toc = time.time()
    print("It takes {:.2f} seconds to finish the second iteration of searching".format(toc - tic))

    # The third iteration of calculation
    tic = time.time()
    M0, delta_M, chi2 = getContrastFromProbability(probability=probability,
                                                   kMean=kMean,
                                                   nroi=nroi,
                                                   Mmin=max(1, M0 - 0.2),
                                                   Mmax=M0 + 0.2,
                                                   Mnum=50)
    toc = time.time()
    print("It takes {:.2f} seconds to finish the third iteration of searching".format(toc - tic))

    print("The contrast is estimated to be" +
          fancyFont.BOLD + fancyFont.GREEN + "{:.2e} +- {:.2e}".format(1 / M0,
                                                                       0.5 / (M0 - delta_M) - 0.5 / (M0 + delta_M)) +
          fancyFont.END
          )
    return M0, delta_M, chi2


def getContrast_CC_VCC_Both(probability_CC, kMean_CC,
                            probability_VCC, kMean_VCC,
                            probability_Both, kMean_Both,
                            nroi):
    print("Processing the photon count probability data for " + fancyFont.CYAN + fancyFont.BOLD + "CC"
          + fancyFont.END)
    contrastResult_CC = getContrast(probability_CC, kMean_CC, nroi)

    print("Processing the photon count probability data for " + fancyFont.CYAN + fancyFont.BOLD + "VCC"
          + fancyFont.END)
    contrastResult_VCC = getContrast(probability_VCC, kMean_VCC, nroi)

    print("Processing the photon count probability data for " + fancyFont.CYAN + fancyFont.BOLD + "Both"
          + fancyFont.END)
    contrastResult_Both = getContrast(probability_Both, kMean_Both, nroi)

    return contrastResult_CC, contrastResult_VCC, contrastResult_Both


def getISF(probability_CC, kMean_CC, probability_VCC, kMean_VCC, probability_Both, kMean_Both, nroi, effectiveOverlap):
    # Get the contrast info for each individual cases
    (contrastResult_CC,
     contrastResult_VCC,
     contrastResult_Both) = getContrast_CC_VCC_Both(probability_CC=probability_CC,
                                                    kMean_CC=kMean_CC,
                                                    probability_VCC=probability_VCC,
                                                    kMean_VCC=kMean_VCC,
                                                    probability_Both=probability_Both,
                                                    kMean_Both=kMean_Both,
                                                    nroi=nroi)

    # Convert the mode number to beta
    beta_CC = 1. / contrastResult_CC[0]
    beta_VCC = 1. / contrastResult_VCC[0]
    beta_Both = 1. / contrastResult_Both[0]

    # Get branching ratio
    ratio = np.mean(kMean_CC) / np.mean(kMean_CC + kMean_VCC)

    # Combine to get the ISF
    isf = beta_Both - ratio ** 2 * beta_CC - (1 - ratio) ** 2 * beta_VCC
    isf /= 2 * ratio * (1 - ratio) * effectiveOverlap * min(beta_CC, beta_VCC)

    return isf, contrastResult_CC, contrastResult_VCC, contrastResult_Both


def getEffectiveOverlap(probability_CC, kMean_CC, probability_VCC, kMean_VCC, probability_Both, kMean_Both, nroi):
    effectiveOverlap, _, _, _ = getISF(probability_CC=probability_CC,
                                       kMean_CC=kMean_CC,
                                       probability_VCC=probability_VCC,
                                       kMean_VCC=kMean_VCC,
                                       probability_Both=probability_Both,
                                       kMean_Both=kMean_Both,
                                       nroi=nroi,
                                       effectiveOverlap=1.0)
    return effectiveOverlap
