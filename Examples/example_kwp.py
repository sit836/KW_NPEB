import sys

sys.path.append("..")

from kw_mle import KWMLE
from plot import *
from simulation import simulate_discrete_prior

data = simulate_discrete_prior(sz=300)

kw_mle = KWMLE(data)
prior, mixture = kw_mle.kw_primal()

plot_prior(kw_mle)
plot_mixture(kw_mle)
