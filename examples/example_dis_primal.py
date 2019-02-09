import sys

sys.path.append("..")

from kw_mle import KWMLE
from plot import *
from simulation import simulation

data = simulation(sz=300, method="dis_uniform")

kw_mle = KWMLE(data)
prior, mixture = kw_mle.kw_primal()

plot_prior(kw_mle)
plot_mixture(kw_mle)
plot_prediction(kw_mle, data)