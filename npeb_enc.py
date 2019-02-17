import numpy as np
import pandas as pd

from kw_mle import KWMLE


class NPEBEnc:
    def __init__(self):
        self.enc_dicts = {}

    def fit(self, categorical_features, label, target_name):
        """
        Fit OneHotEncoder to categorical_features.
        """
        df = categorical_features.join(label)

        for feature in categorical_features.columns:
            target_grouped = df.groupby(feature)[target_name]

            counts_grouped = target_grouped.count().values.flatten()
            means_grouped = target_grouped.mean().values.flatten()
            stds_grouped = target_grouped.std().values.flatten() / np.sqrt(counts_grouped)

            kw_mle = KWMLE(means_grouped, stds=stds_grouped)
            kw_mle.fit()

            enc = kw_mle.prediction(means_grouped, stds=stds_grouped)
            self.enc_dicts[feature] = dict(zip(target_grouped.mean().index, enc))

    def transform(self, categorical_features):
        transformed_features = pd.DataFrame(index=categorical_features.index)
        for feature in categorical_features.columns:
            transformed_features[feature + '_enc'] = categorical_features[feature].map(self.enc_dicts[feature])
        return transformed_features
