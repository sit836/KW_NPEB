import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal

from kw_mle.kw_mle import KWMLE

sz = 1000
data = normal(loc=10, size=sz) + normal(size=sz)
stds = [1]*len(data)
kw_mle = KWMLE(data, stds=stds)
kw_mle.fit()
pred = kw_mle.prediction(np.sort(data), stds) # for visualization, data is sorted

plt.plot(kw_mle.grid_of_mean, kw_mle.prior)
plt.title("Estimated prior")
plt.show()

mixture_sorted = kw_mle.mixture[np.argsort(kw_mle.data)]
plt.plot(np.sort(data), mixture_sorted, '-o')
plt.title("Estimated mixture")
plt.show()

plt.plot(np.sort(data), pred, '-o')
plt.title("Prediction")
plt.show()
