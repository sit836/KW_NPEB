import matplotlib.pyplot as plt

def plot_prior(grid_of_mean, prior):
    plt.plot(grid_of_mean, prior)
    plt.ylabel("Estimated prior")
    plt.show()
