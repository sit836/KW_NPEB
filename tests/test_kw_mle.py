from nose.tools import *
from numpy.random import normal

from kw_mle import *


def generate_data(sz, mean_signal, mean_prior, std_signal, std_prior):
    signal = normal(loc=mean_signal, scale=std_signal, size=sz)
    noise = normal(loc=mean_prior, scale=std_prior, size=sz)
    return signal, signal + noise


def normal_bayes_estimators(data, mean_prior, std_signal, std_prior):
    return ((std_prior ** 2 / (std_prior ** 2 + std_signal ** 2)) * data
            + (std_signal ** 2 / (std_prior ** 2 + std_signal ** 2)) * mean_prior)


def test_kw_primal():
    sz = 100
    mean_signal, mean_prior = 10, 0
    std_signal, std_prior = 1, 1

    signal, data = generate_data(sz, mean_signal, mean_prior,
                                 std_signal, std_prior)
    oracle_pred = normal_bayes_estimators(data, mean_prior, std_signal, std_prior)
    kw_mle = KWMLE(data, stds=[std_prior] * len(data))
    prior, mixture = kw_mle.kw_primal()
    kw_pred = kw_mle.prediction(data, [std_prior] * len(data))

    import matplotlib.pyplot as plt
    plt.scatter(oracle_pred, kw_pred)
    plt.show()
