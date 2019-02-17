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

    train_enc_raw = kw_mle.prediction(train_data, stds=train_stds)
    train_enc = pd.DataFrame(train_enc_raw, index=means_grouped.index, columns=[feature + '_enc'])

    print("train_enc\n", train_enc)

    test_data = pd.DataFrame(X_test[feature], index=X_test[feature], columns=[feature])
    test_data.join(train_enc)
    print("test_data.join(train_enc)\n", test_data.join(train_enc))

