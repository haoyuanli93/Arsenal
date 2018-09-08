import numpy as np
from scipy import stats


#####################################################################################
# Calculate correlations
#####################################################################################
def correlation_n_to_1(n_instance, one_instance):
    """
    Calculate the correlation of n vectors and a single vector.

    :param n_instance: A numpy array of the shape [n,m] where m is the dimension of the
                        vectors.
    :param one_instance: A numpy array of the shape [m] where m is the dimension of the
                        vector.
    :return: A numpy array of the shape [n] which the correlation coefficient.
    """

    # Get mean value
    mean_n = np.mean(n_instance, axis=-1)
    mean_1 = np.mean(one_instance)

    # Shift to zero
    n_instance -= mean_n
    one_instance -= mean_1

    # Get length
    length_n = np.sqrt(np.sum(np.square(n_instance), axis=-1))
    length_1 = np.sqrt(np.sum(np.square(one_instance)))

    # Normalize the vectors
    n_instance /= length_n
    one_instance /= length_1

    # Get inner product
    inner = np.dot(n_instance, one_instance)

    return inner


def correlation_n_to_m(n_instance, m_instance):
    """
    Calculate the correlation of n vectors and a single vector.

    :param n_instance: A numpy array of the shape [n, p] where p is the dimension of the
                        vectors.
    :param m_instance: A numpy array of the shape [m, p] where p is the dimension of the
                        vector.
    :return: A numpy array of the shape [n, m] which the correlation coefficient matrix.
    """
    # Get mean value
    mean_n = np.mean(n_instance, axis=-1)
    mean_m = np.mean(m_instance)

    # Shift to zero
    n_instance -= mean_n
    m_instance -= mean_m

    # Get length
    length_n = np.sqrt(np.sum(np.square(n_instance), axis=-1))
    length_m = np.sqrt(np.sum(np.square(m_instance), axis=-1))

    # Normalize the vectors
    n_instance /= length_n
    m_instance /= length_m

    # Get inner product
    inner = np.matmul(n_instance, m_instance.T)

    return inner


#####################################################################################
# Statistical Distances
#####################################################################################

def js_distance(p, q, base=np.e):
    """
    Calculate the Jensen-Shannon distance for tow distribution p,q for a specific base

    :param p: 1-D numpy array which should have the same shape as q. np.sum(p) should be 1.
    :param q: 1-D numpy array which should have the same shape as p. np.sum(q) should be 1.
    :param base: The base value for the logarithm
    :return: The Jensen-Shannon distance. Notice that this is a metric which means it's the
    square root of the Jensen-Shannon divergence.
    """
    '''
        Implementation of pairwise `jsd` based on  
        https://gist.github.com/zhiyzuo/f80e2b1cfb493a5711330d271a228a3d
    '''

    m = 1. / 2 * (p + q)
    return np.sqrt(stats.entropy(p, m, base=base) / 2. +
                   stats.entropy(q, m, base=base) / 2.)


def js_distance_safe(p, q, base=np.e):
    """
    Calculate the Jensen-Shannon distance for tow distribution p,q for a specific base

    :param p: 1-D numpy array which should have the same shape as q. np.sum(p) can be any positive number.
    :param q: 1-D numpy array which should have the same shape as p. np.sum(q) can be any positive number.
    :param base: The base value for the logarithm
    :return: The Jensen-Shannon distance. Notice that this is a metric which means it's the
    square root of the Jensen-Shannon divergence.
    """
    # convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    # normalize p, q to probabilities
    p, q = p / p.sum(), q / q.sum()

    return js_distance(p=p, q=q, base=base)


def js_distance_batch(p, q, base=np.e):
    """
    Calculate the Jensen-Shannon distance between each pair of distributions in 2D arrays p and q.

    :param p: 2D numpy array. The dimension 1 should be a probability distribution. i.e. np.sum(p, axis=1) should be
                a 1D array of 1s.
    :param q: 2D numpy array. The dimension 1 should be a probability distribution. i.e. np.sum(p, axis=1) should be
                a 1D array of 1s.
    :param base: The base value for the logarithm
    :return: 1D numpy array for the Jensen-Shannon distance for each pairs of distributions in p and q.
    """

    # First step: check the shape
    if len(p.shape) != 2 or len(q.shape) != 2:
        raise Exception("Both p and q has to be 2D numpy arrays where p[i,:] is a probability distribution for "
                        "each i and q[j,:] is a probability for each j.")

    if p.shape != q.shape:
        raise Exception("The shape of p and q has to be the same.")

    # Second step: Check if the summation is 1 for each rows.
    holder = np.ones(p.shape[0])

    if not (np.allclose(np.sum(p, axis=1), holder, rtol=1e-8, atol=1e-10) and
            np.allclose(np.sum(q, axis=1), holder, rtol=1e-8, atol=1e-10)):
        raise Exception("Both np.sum(p, axis=1) and np.sum(q, axis=1) should be 1D array of 1s.")

    # Third step: Calculate the distance
    for l in range(p.shape[0]):
        holder[l] = js_distance(p[l], q[l], base=base)

    return holder
