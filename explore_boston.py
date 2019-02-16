import pandas as pd
from sklearn.datasets import load_boston

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

# cat_features = ['ZN', 'CHAS', 'RAD']
cat_features = ['ZN']

target = 'PRICE'

feature = cat_features
target_grouped = df.groupby(feature)[target]

print(df[feature].unique())
gp_stds = pd.DataFrame(df.groupby(feature)[target].std())

# print("df[feature].unique(): ", pd.Series(df[feature]).unique())
# print("gp_stds: ", gp_stds)
#
# std_col = feature + '_GP_STDS'
# gp_stds.columns = [std_col]
# df_joined = df.set_index(feature).join(gp_stds)
#
# obs, stds = df_joined[target].values, df_joined[std_col].values
# kw_mle = KWMLE(obs, stds=stds)
# prior, mixture = kw_mle.kw_dual()
# pred = kw_mle.prediction(obs, stds=stds)
# df[feature] = pred

# for feature in cat_features:
#     target_grouped = df.groupby(feature)[target]
#
#     gp_stds = pd.DataFrame(df.groupby(feature)[target].std())
#     std_col = feature + '_GP_STDS'
#     gp_stds.columns = [std_col]
#     df_joined = df.set_index(feature).join(gp_stds)
#
#     obs, stds = df_joined[target].values, df_joined[std_col].values
#     kw_mle = KWMLE(obs, stds=stds)
#     prior, mixture = kw_mle.kw_dual()
#     pred = kw_mle.prediction(obs, stds=stds)
#     df[feature] = pred
