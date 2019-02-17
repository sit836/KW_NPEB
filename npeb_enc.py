from kw_mle import KWMLE


class NPEBEnc:
    def __init__(self, categorical_features=None):
        self.categorical_features = categorical_features

    def fit(self, categorical_features, label, target_name):
        """
        Fit OneHotEncoder to categorical_features.
        """
        for feature in cat_features:
            target_grouped = train.groupby(feature)[target_name]

            counts_grouped = target_grouped.count().values.flatten()
            means_grouped = target_grouped.mean().values.flatten()
            stds_grouped = target_grouped.std().values.flatten() / np.sqrt(counts_grouped)

            kw_mle = KWMLE(means_grouped, stds=stds_grouped)
            kw_mle.fit()

            enc = kw_mle.prediction(means_grouped, stds=stds_grouped)
            enc_dict = dict(zip(target_grouped.mean().index, enc))
