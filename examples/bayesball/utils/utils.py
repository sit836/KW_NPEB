import numpy as np


def get_train_test(df_raw):
    MIN_NUM_AB = 10
    ab_s1 = df_raw.iloc[:, [4, 5, 6]]
    s1_bool = ab_s1.sum(axis=1) > MIN_NUM_AB
    train_df = df_raw[s1_bool]

    ab_s2 = df_raw.iloc[:, [7, 8, 9]]
    s2_bool = ab_s2.sum(axis=1) > MIN_NUM_AB
    test_df = df_raw[s1_bool & s2_bool]

    train = train_df[['First Name', 'Last Name', 'AB(4)', 'AB(5)', 'AB(6)', 'H(4)', 'H(5)', 'H(6)']].copy()
    test = test_df[['First Name', 'Last Name', 'AB(7)', 'AB(8)', 'AB(9-10)', 'H(7)', 'H(8)', 'H(9-10)']].copy()

    train['AB'] = train[['AB(4)', 'AB(5)', 'AB(6)']].sum(axis=1)
    train['H'] = train[['H(4)', 'H(5)', 'H(6)']].sum(axis=1)
    test['AB'] = test[['AB(7)', 'AB(8)', 'AB(9-10)']].sum(axis=1)
    test['H'] = test[['H(7)', 'H(8)', 'H(9-10)']].sum(axis=1)

    return train, test


def variance_stabilizing(num_hit, num_ab):
    """
    A variance stabilizing transformation stated in Brown (2008)
    """
    return np.arcsin(np.sqrt((num_hit + 1 / 4) / (num_ab + 1 / 2)))


def tse(truth, pred, std):
    """
    Total squared error (Brown, 2008)
    """
    return sum((truth - pred) ** 2) - sum(std ** 2)


def james_stein_prediction(y, std):
    """
    Heterogeneous James Stein estimator (Brown, 2008, page 20)
    """
    w = (1 / (std ** 2)) / sum((1 / std) ** 2)
    wlse = [sum(w * y)] * len(y)
    S = sum(((y - wlse) / std) ** 2)
    return wlse + (1 - (len(y) - 3) / S) * (y - wlse)
