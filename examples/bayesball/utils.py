import numpy as np


def get_train_test(df_raw):
    MIN_NUM_AB = 10
    ab_s1 = df_raw.iloc[:, [4, 5, 6]]
    s1_bool = ab_s1.sum(axis=1) > MIN_NUM_AB
    train_df = df_raw[s1_bool]

    ab_s2 = df_raw.iloc[:, [7, 8, 9]]
    s2_bool = ab_s2.sum(axis=1) > MIN_NUM_AB
    test_df = df_raw[s1_bool & s2_bool]

    train = train_df[['First Name', 'Last Name', 'AB(4)', 'AB(5)', 'AB(6)', 'H(4)', 'H(5)', 'H(6)']]
    test = test_df[['First Name', 'Last Name', 'AB(7)', 'AB(8)', 'AB(9-10)', 'H(7)', 'H(8)', 'H(9-10)']]

    train['AB'] = train[['AB(4)', 'AB(5)', 'AB(6)']].sum(axis=1)
    train['H'] = train[['H(4)', 'H(5)', 'H(6)']].sum(axis=1)
    test['AB'] = test[['AB(7)', 'AB(8)', 'AB(9-10)']].sum(axis=1)
    test['H'] = test[['H(7)', 'H(8)', 'H(9-10)']].sum(axis=1)

    return train, test


def variance_stabilizing(num_hit, num_ab):
    """
    A variance stabilizing transformation stated in Brown (2008)
    :param num_hit:
    :param num_ab:
    :return:
    """
    return np.arcsin(np.sqrt((num_hit + 1 / 4) / (num_ab + 1 / 2)))
