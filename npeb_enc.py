import numpy as np

from kw_mle import KWMLE


class NPEBEnc:
    def __init__(self, categorical_features=None):
        self.categorical_features = categorical_features

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
            self.enc_dict = dict(zip(target_grouped.mean().index, enc))

    def transform(self, categorical_features):
        transformed_features = pd.DataFrame(index=categorical_features.index)
        for feature in categorical_features.columns:
            transformed_features[feature + '_enc'] = categorical_features[feature].map(self.enc_dict)
        return transformed_features


import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
df = pd.DataFrame(boston.data)
df.columns = boston.feature_names
df['PRICE'] = boston.target

cat_names = ['CHAS', 'RAD']
target_name = 'PRICE'

X_train, X_test, y_train, y_test = train_test_split(df[cat_names], df[target_name], test_size=0.50, random_state=123)
train = X_train.join(y_train)

enc = NPEBEnc()
enc.fit(X_train, y_train, target_name)
print(enc.transform(X_test))
