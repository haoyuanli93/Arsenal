import numpy as np
from numba import jit


@jit(paralle=True)
def get_chi2(y1, x1, y2, x2, y3, x3, y4, x4,
             sigma_y1, sigma_x1,
             sigma_y2, sigma_x2,
             sigma_y3, sigma_x3,
             sigma_y4, sigma_x4,
             a1, a2, a3, a4, b):
    chi2 = np.divide((y1 - a1 * x1 - b) ** 2,
                     (a1 * sigma_x1) ** 2 + sigma_y1 ** 2)

    chi2 += np.divide((y2 - a2 * x2 - b) ** 2,
                      (a2 * sigma_x2) ** 2 + sigma_y2 ** 2)

    chi2 += np.divide((y3 - a3 * x3 - b) ** 2,
                      (a3 * sigma_x3) ** 2 + sigma_y3 ** 2)

    chi2 += np.divide((y4 - a4 * x4 - b) ** 2,
                      (a4 * sigma_x4) ** 2 + sigma_y4 ** 2)

    return chi2


def get_MC_guess(a1_range, a2_range, a3_range, a4_range, b_range, num_of_trial,
                 y1, x1, y2, x2, y3, x3, y4, x4,
                 sigma_y1, sigma_x1,
                 sigma_y2, sigma_x2,
                 sigma_y3, sigma_x3,
                 sigma_y4, sigma_x4,
                 ):
    # Get random guess of the parameters
    a1_list = np.random.rand(num_of_trial) * (a1_range[1] - a1_range[0]) + a1_range[0]
    a2_list = np.random.rand(num_of_trial) * (a2_range[1] - a2_range[0]) + a2_range[0]
    a3_list = np.random.rand(num_of_trial) * (a3_range[1] - a3_range[0]) + a3_range[0]
    a4_list = np.random.rand(num_of_trial) * (a4_range[1] - a4_range[0]) + a4_range[0]

    b_list = np.random.rand(num_of_trial) * (b_range[1] - b_range[0]) + b_range[0]

    chi_list = np.zeros(num_of_trial)

    for idx in range(num_of_trial):
        chi_list[idx] = get_chi2(y1=y1,
                                 x1=x1,
                                 y2=y2,
                                 x2=x2,
                                 y3=y3,
                                 x3=x3,
                                 y4=y4,
                                 x4=x4,
                                 sigma_y1=sigma_y1,
                                 sigma_x1=sigma_x1,
                                 sigma_y2=sigma_y2,
                                 sigma_x2=sigma_x2,
                                 sigma_y3=sigma_y3,
                                 sigma_x3=sigma_x3,
                                 sigma_y4=sigma_y4,
                                 sigma_x4=sigma_x4,
                                 a1=a1_list[idx],
                                 a2=a2_list[idx],
                                 a3=a3_list[idx],
                                 a4=a4_list[idx],
                                 b=b_list[idx])

    return chi_list, a1_list, a2_list, a3_list, a4_list, b_list

