from numpy.random import normal

from kw_mle import KWMLE
from plot import *

sz = 300
data = normal(loc=10, size=sz) + normal(size=sz)
stds = [1]*len(data)
kw_mle = KWMLE(data, stds=stds)
prior, mixture = kw_mle.kw_dual()

plot_prior(kw_mle)
plot_mixture(kw_mle)
plot_prediction(kw_mle, data, stds)
