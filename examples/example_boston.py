import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from npeb_enc import NPEBEnc

def hist_per_cat(df, cat_names, target_name):
    for feature in cat_names:
        target_grouped = df.groupby(feature)[target_name]
        ncols = 2
        nrows = int(np.ceil(target_grouped.ngroups / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        for (key, ax) in zip(target_grouped.groups.keys(), axes.flatten()):
            target_grouped.get_group(key).hist(ax=ax)
            ax.set_title(f"{feature}_{key}")
        ax.legend()
    fig.subplots_adjust(hspace=0.5)
    plt.show()

boston = load_boston()
df = pd.DataFrame(boston.data)
df.columns = boston.feature_names
df['PRICE'] = boston.target
cat_names = ['CHAS', 'RAD']
target_name = 'PRICE'

hist_per_cat(df, cat_names, target_name)

X_train, X_test, y_train, y_test = train_test_split(df[cat_names], df[target_name], test_size=0.50, random_state=123)
enc = NPEBEnc()
enc.fit(X_train, y_train, target_name)
print(enc.transform(X_test))
