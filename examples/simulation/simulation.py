"""
Data is assumed to the sum of the true signal and Gaussian noise.
"""
import numpy as np
from numpy.random import normal

from plot import *
from kw_mle import KWMLE


def simulation(sz, method):
    if method == "dis_uniform":
        return simulate_discrete_uniform_prior(sz)
    elif method == "norm":
        return simulate_norm_prior(sz)
    else:
        raise ValueError("The type of prior can only be dis_uniform or norm.")


def simulate_discrete_uniform_prior(sz):
    support_prior = [7, 10]
    signal = np.random.choice(support_prior, sz)
    return signal + normal(size=sz)


def simulate_norm_prior(sz):
    signal = normal(loc=10, size=sz)
    return signal + normal(size=sz)

sz = 1000
data = normal(loc=10, size=sz) + normal(size=sz)
stds = [1]*len(data)
kw_mle = KWMLE(data, stds=stds)
kw_mle.fit()

plot_prior(kw_mle)
plot_mixture(kw_mle)
plot_prediction(kw_mle, data, stds)
