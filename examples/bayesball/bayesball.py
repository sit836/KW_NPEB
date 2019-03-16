DATA_PATH = 'D:/py_projects/npeb/MAP/data/brown_2008.csv'

import numpy as np
import pandas as pd

from utils import *

df_raw = pd.read_csv(DATA_PATH)
train, test = get_train_test(df_raw)

X = train[['AB(4)', 'AB(5)', 'AB(6)', 'H(4)', 'H(5)', 'H(6)']]
train_label = variance_stabilizing(train['H'], train['AB'])
test_label = variance_stabilizing(test['H'], test['AB'])

GRID_SZ = 400
grid_theta = np.linspace(min(train_label) - 1, max(train_label) + 1, GRID_SZ)

std_train = 1/(2*np.sqrt(train['AB'].values))
std_test = 1/(2*np.sqrt(test['AB'].values))

dual = kwd(train_label, std_train, grid_theta)
prior = dual['p']

naive_pred_raw = train_label
npeb_pred_raw = npeb_prediction(train_label, std_train, prior, grid_theta)
js_pred_raw = james_stein_prediction(train_label, std_train)

naive_pred = naive_pred_raw.loc[test_label.index]
npeb_pred = npeb_pred_raw.loc[test_label.index]
js_pred = js_pred_raw.loc[test_label.index]

tse_naive = tse(test_label, naive_pred, std_test)
tse_npeb = tse(np.array(test_label), np.array(npeb_pred).flatten(), std_test)
tse_js = tse(np.array(test_label), np.array(js_pred).flatten(), std_test)
tse_naive, tse_npeb, tse_js
