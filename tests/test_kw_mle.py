import numpy as np
from nose.tools import *
from numpy.random import normal

from kw_mle import *


def generate_data(sz, mean_signal, mean_prior, std_signal, std_prior):
    signal = normal(loc=mean_signal, scale=std_signal, size=sz)
    noise = normal(loc=mean_prior, scale=std_prior, size=sz)
    return signal, signal + noise


def Bayes_oracle_estimator(data, std_signal, mean_prior, std_prior):
    """
    See Large-scale inference: empirical Bayes methods for estimation, testing, and prediction (Efron, 2010, page 7)
    """
    return mean_prior + (std_prior**2/(std_prior**2+std_signal**2))*(data-mean_prior)

def James_Stein_estimator(data, std_signal):
    """
    See Large-scale inference: empirical Bayes methods for estimation, testing, and prediction (Efron, 2010, page 7)
    """
    mean_data = np.mean(data)
    dev = sum((data - mean_data)**2)
    return mean_data + (1 - (len(data)-3)*(std_signal**2)/dev)*(data - mean_data)

def test_kw_primal():
    sz = 500
    mean_signal, mean_prior = 10, 0
    std_signal, std_prior = 1, 1

    signal, data = generate_data(sz, mean_signal, mean_prior,
                                 std_signal, std_prior)
    oracle_pred = Bayes_oracle_estimator(data, std_signal, mean_prior, std_prior)
    js_pred = James_Stein_estimator(data, std_signal)
    kw_mle = KWMLE(data, stds=[std_prior] * len(data))
    _, _ = kw_mle.kw_primal()
    kw_pred = kw_mle.prediction(data, [std_prior] * len(data))

    import matplotlib.pyplot as plt
    # plt.scatter(js_pred, kw_pred)
    plt.scatter(data, js_pred)
    plt.scatter(data, oracle_pred)
    plt.show()

    # plt.plot(kw_mle.grid_of_mean, kw_mle.prior)
    # plt.title("Estimated prior")
    # plt.show()

    print(sum((oracle_pred - signal) ** 2), sum((js_pred - signal)**2), sum((kw_pred - signal)**2))
