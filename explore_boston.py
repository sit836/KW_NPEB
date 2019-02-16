import pandas as pd
from sklearn.datasets import load_boston

from kw_mle import KWMLE

boston = load_boston()
df = pd.DataFrame(boston.data)
df.columns = boston.feature_names
df['PRICE'] = boston.target

print(df.nunique())

target = 'PRICE'
feature = 'RAD'
target_grouped = df.groupby(feature)[target]

# ncols = 2
# nrows = int(np.ceil(target_grouped.ngroups / ncols))
# fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
# for (key, ax) in zip(target_grouped.groups.keys(), axes.flatten()):
#     target_grouped.get_group(key).hist(ax=ax)
# ax.legend()
# plt.show()

gp_stds = pd.DataFrame(df.groupby(feature)[target].std())
gp_stds.columns = [feature + '_GP_STDS']
df_joined = df.set_index(feature).join(gp_stds)

kw_mle = KWMLE(df_joined[target].values, stds=df_joined[feature + '_GP_STDS'].values)
prior, mixture = kw_mle.kw_dual()
pred = kw_mle.prediction(df_joined[target].values, stds=df_joined[feature + '_GP_STDS'].values)
