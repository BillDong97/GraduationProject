
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt


doData = pd.read_excel('除氧床.xlsx', header=0)
time1 = np.array(doData["运行时长"])
x1 = np.array(doData.iloc[:, 2:7], dtype='float32')
y1 = np.array(doData['出水'], dtype='float32')

mixedData = pd.read_excel('混床.xlsx', header=0)
time2 = np.array(mixedData["运行时长"])
x2 = np.array(mixedData.iloc[:, 2:], dtype='float32')
# x2 = np.array(mixedData.iloc[:, 3], dtype='float32')
y2 = np.array(mixedData['累积水量'], dtype='float32')


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
