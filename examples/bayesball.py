DATA_PATH = 'D:/py_projects/npeb/MAP/data/brown_2008.csv'

import numpy as np
import pandas as pd
from kw_mle import KWMLE
from preprocess import preprocessing
from primal import kwp


df_raw = pd.read_csv(DATA_PATH)
train, test = preprocessing(df_raw)

train_x = train[['AB(4)', 'AB(5)', 'AB(6)', 'H(4)', 'H(5)', 'H(6)']]
train_y = transformation(train['H'],train['AB'])
test_y = transformation(test['H'],test['AB'])

GRID_SZ = 400
grid_theta = np.linspace(min(train_y)-1,max(train_y)+1,GRID_SZ)

std_train = 1/(2*np.sqrt(train['AB'].values))
std_test = 1/(2*np.sqrt(test['AB'].values))

dual = kwd(train_y,std_train, grid_theta)
prior = dual['p']

naive_pred_raw = train_y
npeb_pred_raw = npeb_prediction(train_y, std_train, prior, grid_theta)
js_pred_raw = james_stein_prediction(train_y, std_train)

naive_pred = naive_pred_raw.loc[test_y.index]
npeb_pred = npeb_pred_raw.loc[test_y.index]
js_pred = js_pred_raw.loc[test_y.index]

rf = RandomForestRegressor(n_estimators = 100, random_state=123)
rf.fit(train_y.index.values.reshape(-1,1), train_y.values.reshape(-1,1))
rf_pred = rf.predict(test_y.index.values.reshape(-1,1))

rf_full = RandomForestRegressor(n_estimators = 100, random_state=123, max_features='sqrt', max_depth=1, min_samples_leaf=10)
rf_full.fit(train_x, train_y)
rf_full_pred = rf_full.predict(train_x.loc[test_y.index])

tse_naive = tse(test_y,naive_pred,std_test)
tse_npeb = tse(np.array(test_y),np.array(npeb_pred).flatten(),std_test)
tse_js = tse(np.array(test_y),np.array(js_pred).flatten(),std_test)
tse_rf = tse(test_y,rf_pred,std_test)
tse_rf_full = tse(test_y,rf_full_pred,std_test)
tse_naive, tse_npeb, tse_js,  tse_rf, tse_rf_full
