from scipy import optimize
from numba import jit


@jit
def linear(x, a, b):
    """
    A linear function.
    :param x: Argument
    :param a: coefficient
    :param b: constant
    :return: ax + b
    """
    return a * x + b


def fit_for_linear_function(x_data, y_data, a_init=0., b_init=0.,
                            a_range=(-10., 10.), b_range=(-10., 10.), covariance=True):
    """
    Fit for a linear function.

    :param x_data: The x data
    :param y_data: The y data
    :param a_init: The initial value of a.
    :param b_init: The initial value of b.
    :param a_range: The range of a.
    :param b_range: The range of b.
    :param covariance: If true, return the covariance, otherwise, do not return the covariance.
    :return: parameters, parameter_convariance.
    """
    params, params_covariance = optimize.curve_fit(linear, x_data, y_data,
                                                   p0=(a_init, b_init),
                                                   method="dogbox",
                                                   bounds=([b_range[0],
                                                            a_range[0]],
                                                           [b_range[1],
                                                            a_range[1]]))
    if covariance:
        return params, params_covariance
    else:
        return params
