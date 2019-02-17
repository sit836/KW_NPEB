from numpy.random import normal

from old.kw_mle import *


def generate_data(sz, mean_signal, std_signal, std_noise):
    signal = normal(loc=mean_signal, scale=std_signal, size=sz)
    noise = normal(loc=0, scale=std_noise, size=sz)
    return signal, signal + noise


def Bayes_oracle_estimator(data, mean_signal, std_signal, std_noise):
    """
    See Large-scale inference: empirical Bayes methods for estimation, testing, and prediction (Efron, 2010, page 6)
    """
    return mean_signal + (std_signal ** 2 / (std_signal ** 2 + std_noise ** 2)) * (data - mean_signal)


def James_Stein_estimator(data, std_noise):
    """
    See Large-scale inference: empirical Bayes methods for estimation, testing, and prediction (Efron, 2010, page 6)
    """
    mean_data = np.mean(data)
    dev = sum((data - mean_data) ** 2)
    return mean_data + (1 - (len(data) - 3) * (std_noise ** 2) / dev) * (data - mean_data)


def test_kw_primal():
    sz = 500
    mean_signal = 10
    std_signal, std_noise = 1, 1

    signal, data = generate_data(sz, mean_signal,
                                 std_signal, std_noise)
    oracle_pred = Bayes_oracle_estimator(data, mean_signal, std_signal, std_noise)
    js_pred = James_Stein_estimator(data, std_noise)
    kw_mle = KWMLE(data, stds=[std_noise] * len(data))
    _, _ = kw_mle.kw_primal()
    kw_pred = kw_mle.prediction(data, [std_noise] * len(data))

    print("Oracel, James-Stein, Kiefer-Wolfowitz")
    print(sum((oracle_pred - signal) ** 2), sum((js_pred - signal) ** 2), sum((kw_pred - signal) ** 2))
