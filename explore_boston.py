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

X_train, X_test, y_train, y_test = train_test_split(df[cat_features], df[target], test_size=0.33, random_state=123)
train = X_train.join(y_train)

for feature in cat_features:
    target_grouped = train.groupby(feature)[target]

    gp_stds = pd.DataFrame(train.groupby(feature)[target].std())
    std_col = feature + '_GP_STDS'
    gp_stds.columns = [std_col]
    train_joined = train.set_index(feature).join(gp_stds)

    train_obs, train_stds = train_joined[target].values, train_joined[std_col].values

    kw_mle = KWMLE(train_obs, stds=train_stds)
    prior, mixture = kw_mle.kw_dual()
    pred = kw_mle.prediction(train_obs, stds=train_stds)
    train[feature + '_new'] = pred
