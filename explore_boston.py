import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from kw_mle import KWMLE

boston = load_boston()
df = pd.DataFrame(boston.data)
df.columns = boston.feature_names
df['PRICE'] = boston.target

# import matplotlib.pyplot as plt
# feature, target = 'CHAS', 'PRICE'
# target_grouped = df.groupby(feature)[target]
# ncols = 2
# nrows = int(np.ceil(target_grouped.ngroups / ncols))
# fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
# for (key, ax) in zip(target_grouped.groups.keys(), axes.flatten()):
#     target_grouped.get_group(key).hist(ax=ax)
# ax.legend()
# plt.show()

cat_features = ['CHAS', 'RAD']
target = 'PRICE'

X_train, X_test, y_train, y_test = train_test_split(df[cat_features], df[target], test_size=0.50, random_state=123)
train = X_train.join(y_train)

for feature in cat_features:
    target_grouped = train.groupby(feature)[target]

    counts_grouped = target_grouped.count().values.flatten()
    means_grouped = target_grouped.mean().values.flatten()
    stds_grouped = target_grouped.std().values.flatten() / np.sqrt(counts_grouped)

    kw_mle = KWMLE(means_grouped, stds=stds_grouped)
    kw_mle.fit()

    train_enc = kw_mle.prediction(means_grouped, stds=stds_grouped)
    enc_dict = dict(zip(target_grouped.mean().index, train_enc))
    X_test[feature + '_enc'] = X_test[feature].map(enc_dict)

print(X_test)
