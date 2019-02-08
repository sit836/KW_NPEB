import numpy as np

from kw_mle import kwp
from plot import *
from simulation import simulate_discrete_prior

sample_sz = 100
len_grid = 300

data = simulate_discrete_prior(sample_sz)
grid_of_mean = np.linspace(min(data), max(data), len_grid)

prior, mixture = kwp(data, grid_of_mean)

plot_prior(grid_of_mean, prior)
