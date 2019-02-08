import matplotlib.pyplot as plt
import numpy as np


def plot_prior(grid_of_mean, prior):
    plt.plot(grid_of_mean, prior)
    plt.ylabel("Estimated prior")
    plt.show()


def plot_mixture(grid_of_mean, mixture):
    grid_of_mean_sorted = np.sort(grid_of_mean)
    mixture_sorted = mixture[np.argsort(grid_of_mean)]

    plt.plot(grid_of_mean_sorted, mixture_sorted, '-o')
    plt.ylabel("Estimated mixture")
    plt.show()
