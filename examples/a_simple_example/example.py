from numpy.random import normal

from plot import *
from kw_mle import KWMLE

sz = 1000
data = normal(loc=10, size=sz) + normal(size=sz)
stds = [1]*len(data)
kw_mle = KWMLE(data, stds=stds)
kw_mle.fit()

plot_prior(kw_mle)
plot_mixture(kw_mle)
plot_prediction(kw_mle, data, stds)
