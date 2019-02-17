import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from kw_mle import KWMLE

boston = load_boston()
df = pd.DataFrame(boston.data)
df.columns = boston.feature_names
df['PRICE'] = boston.target

# ncols = 2
# nrows = int(np.ceil(target_grouped.ngroups / ncols))
# fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
# for (key, ax) in zip(target_grouped.groups.keys(), axes.flatten()):
#     target_grouped.get_group(key).hist(ax=ax)
# ax.legend()
# plt.show()

cat_features = ['CHAS', 'RAD']
target = 'PRICE'

feature = cat_features
target_grouped = df.groupby(feature)[target]

X_train, X_test, y_train, y_test = train_test_split(df[cat_features], df[target], test_size=0.50, random_state=123)
train = X_train.join(y_train)

for feature in cat_features:
    target_grouped = train.groupby(feature)[target]

    counts_grouped = pd.DataFrame(target_grouped.count())
    means_grouped = pd.DataFrame(target_grouped.mean())
    stds_grouped_raw = pd.DataFrame(target_grouped.std())
    stds_grouped = stds_grouped_raw / np.sqrt(counts_grouped)
    std_col = feature + '_STDS_GP'
    stds_grouped.columns = [std_col]

    train_data, train_stds = means_grouped.values.flatten(), stds_grouped.values.flatten()
    kw_mle = KWMLE(train_data, stds=train_stds)
    prior, mixture = kw_mle.kw_dual()

    train_enc = kw_mle.prediction(train_data, stds=train_stds)
    enc_dict = dict(zip(means_grouped.index, train_enc))
    X_test[feature + '_enc'] = X_test[feature].map(enc_dict)

print(X_test)
