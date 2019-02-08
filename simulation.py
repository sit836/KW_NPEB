"""
Data is assumed to the sum of the true signal and Gaussian noise.
"""

import numpy as np


def simulate_discrete_prior(sz):
    """
    :param sz: size of data
    :return: simulated data
    """
    support_prior = [5, 15]
    signal = np.random.choice(support_prior, sz)
    return signal + np.random.normal(size=sz)
