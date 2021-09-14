
from data import x2, y2, time2, data_split
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 划分数据集
x_train, y_train, x_test, y_test, t_test = data_split(x2, y2, time2, 0.8)

# 数据标准化
x_scaler = StandardScaler()
y_scaler = StandardScaler()
x = x_scaler.fit_transform(x_train)
y = y_scaler.fit_transform(y_train.reshape(-1, 1))
x_test = x_scaler.fit_transform(x_test)
y_test = y_scaler.fit_transform(y_test.reshape(-1, 1))

# 训练模型
# model = SGDRegressor(loss='huber', fit_intercept=False, max_iter=1000000)
# model.fit(x, y.ravel())
model = LinearRegression()
model.fit(x, y)
print(model.score(x_test, y_test))

y_predict = model.predict(x_test)
plt.figure()
plt.plot(t_test, y_test, label='test')
plt.plot(t_test, y_predict, label='predict')
plt.xlim([t_test[0], t_test[-1]])
plt.legend(loc='upper left')
plt.show()
