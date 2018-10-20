import numpy as np

"""
This module only contains very rarely used function that should not be used outside this package 
since I will not test these functions outside this package at all.
"""


##################################################################################################################
# For geometry
##################################################################################################################
def cummulate_product_with_local_exclusion_dim1(num_dim1,
                                                arry):
    """
    This function calculate the following things

                [ [a11, a12, a13,] ,                 [[a12*a13, a11*a13, a11*a12],
                  [a21, a22, a23,] ,     ==>>         [a22*a23, a21*a23, a21,a22],
                  ...                                   ...
                  [an1, an2, an3]]                     ]

    :param arry:
    :param num_dim1:
    :return:
    """

    cum_prod = np.ones_like(arry)
    # Firstly, one deal with the first entry and the last entry
    cum_prod[:, 0] = np.prod(arry[:, 1:], axis=-1)
    cum_prod[:, num_dim1 - 1] = np.prod(arry[:, :num_dim1 - 1], axis=-1)

    # calculate all the other values
    if num_dim1 >= 3:
        for l in range(1, num_dim1 - 1):
            cum_prod[:, l] = np.multiply(np.prod(arry[:, 0:l], axis=-1),
                                         np.prod(arry[:, l + 1:], axis=-1))

    return cum_prod
