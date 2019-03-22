import numpy as np
import pandas as pd
from utils.utils import *

from kw_mle import KWMLE

DATA_PATH = 'D:/py_projects/npeb/MAP/data/brown_2008.csv'
df_raw = pd.read_csv(DATA_PATH)

train, test = get_train_test(df_raw)
train_label = variance_stabilizing(train['H'].values, train['AB'].values)
test_label = variance_stabilizing(test['H'].values, test['AB'].values)

std_train = 1/(2*np.sqrt(train['AB'].values))
std_test = 1/(2*np.sqrt(test['AB'].values))

kw_mle = KWMLE(train_label, stds=std_train)
kw_mle.fit()

naive_pred_raw = pd.DataFrame(train_label, index=train.index)
npeb_pred_raw = pd.DataFrame(kw_mle.prediction(train_label, std_train), index=train.index)
js_pred_raw = pd.DataFrame(james_stein_prediction(train_label, std_train), index=train.index)

naive_pred = naive_pred_raw.loc[test.index]
npeb_pred = npeb_pred_raw.loc[test.index]
js_pred = js_pred_raw.loc[test.index]

tse_naive = tse(np.array(test_label), np.array(naive_pred).flatten(), std_test)
tse_npeb = tse(np.array(test_label), np.array(npeb_pred).flatten(), std_test)
tse_js = tse(np.array(test_label), np.array(js_pred).flatten(), std_test)

print("relative total squared error of nonparametric empirical Bayes: ", tse_npeb/tse_naive)
print("relative total squared error of James-Stein estimator: ", tse_js/tse_naive)
