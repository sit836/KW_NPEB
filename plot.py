import matplotlib.pyplot as plt
import numpy as np


def plot_prior(kw_mle):
    plt.plot(kw_mle.grid_of_mean, kw_mle.prior)
    plt.title("Estimated prior")
    plt.show()


def plot_mixture(kw_mle):
    df_sorted = np.sort(kw_mle.df)
    mixture_sorted = kw_mle.mixture[np.argsort(kw_mle.df)]

    plt.plot(df_sorted, mixture_sorted, '-o')
    plt.title("Estimated mixture")
    plt.show()

def plot_prediction(kw_mle):
    pred = kw_mle.prediction()

    plt.plot(kw_mle.df, pred)
    plt.title("Prediction")
    plt.show()
