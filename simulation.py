import numpy as np


def simulate_discrete_prior(sz):
    """
    Generate simulated data, where data is assumed to the sum of the true signal
    and white Gaussian noise.
    :param sz: size of data
    :return: simulated data
    """
    support_prior = [8, 12]
    signal = np.random.choice(support_prior, sz)
    return signal + np.random.normal(size=sz)
