import numpy as np
from data import X1, y1, X2, y2, doExpire, mixedExpire
from torch.nn import LSTM
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit()
for train_index, test_index in tscv.split(X1):
    print("TRAIN:", train_index, "\n", "TEST:", test_index)
    X1_train, X1_test = X1[train_index], X1[test_index]
    y1_train, y1_test = y1[train_index], y1[test_index]
    rnn = LSTM(X1_train)

# SGDRegressor(alpha=0.0001, average=False, epsilon=0.1, eta0=0.01,
#              fit_intercept=True, l1_ratio=0.15, learning_rate='invscaling',
#              loss='squared_loss', max_iter=5, penalty='l2',
#              power_t=0.25, random_state=None, shuffle=True, tol=None, verbose=0,
#              warm_start=False)
# sgd1 = SGDRegressor()
# sgd1.fit(X1_train, y1_train)
#
# result1 = sgd1.predict(doExpires)
# print(result1)