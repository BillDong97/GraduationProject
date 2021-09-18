
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt


doData = pd.read_excel('除氧床溶解氧数据集.xlsx', header=0)
time1 = np.array(doData.iloc[:, 0])
x1 = np.array(doData.iloc[:, 1:6], dtype='float32')
y1 = np.array(doData['出水'], dtype='float32')

mixedData = pd.read_excel('混床电导率数据集.xlsx', header=0)
time2 = np.array(mixedData.iloc[:, 0])
x2 = np.array(mixedData.iloc[:, 1:6], dtype='float32')
y2 = np.array(mixedData.iloc[:, 6], dtype='float32')

SmallMixedData = pd.read_excel('小装置混床.xlsx', header=0)
time3 = np.array(SmallMixedData.iloc[:, 0])
x3 = np.array(SmallMixedData.iloc[:, 1:9], dtype='float32')
y3 = np.array(SmallMixedData.iloc[:, 9:], dtype='float32')


def data_split(x, y, time, train_data_ratio):
    data_len = len(x)
    train_len = int(train_data_ratio * data_len)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]
    t_test = time[train_len:]

    return x_train, y_train, x_test, y_test, t_test


if __name__ == "__main__":
    print(x2)
    # fig1 = plt.figure(figsize=(20, 15))
    # plt.plot(time1, doData["出水"])
    # plt.plot(time1, doData["4L"])
    # plt.plot(time1, doData["3L"])
    # plt.plot(time1, doData["2L"])
    # plt.show()
    # fig2 = plt.figure(figsize=(20, 15))
    # plt.plot(time2, mixedData["出水"])
    # plt.plot(time2, mixedData["4L"])
    # plt.plot(time2, mixedData["3L"])
    # plt.plot(time2, mixedData["2L"])
    # plt.plot(time2, mixedData["进水"])
    # plt.show()
