import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal

from kw_mle import KWMLE


def plot_prior(kw_mle):
    plt.plot(kw_mle.grid_of_mean, kw_mle.prior)
    plt.title("Estimated prior")
    plt.show()


def plot_mixture(kw_mle):
    data_sorted = np.sort(kw_mle.data)
    mixture_sorted = kw_mle.mixture[np.argsort(kw_mle.data)]

    plt.plot(data_sorted, mixture_sorted, '-o')
    plt.title("Estimated mixture")
    plt.show()


def plot_prediction(kw_mle, data, stds):
    data_sorted = np.sort(data)
    pred = kw_mle.prediction(data_sorted, stds)

    plt.plot(data_sorted, pred, '-o')
    plt.title("Prediction")
    plt.show()

sz = 1000
data = normal(loc=10, size=sz) + normal(size=sz)
stds = [1]*len(data)
kw_mle = KWMLE(data, stds=stds)
kw_mle.fit()

plot_prior(kw_mle)
plot_mixture(kw_mle)
plot_prediction(kw_mle, data, stds)
