import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


doData = pd.read_excel('除氧床.xlsx', header=0)
time1 = doData["运行时长"]
X1 = np.array(doData.iloc[:, 2:7], dtype='float32')
y1 = np.array(doData['出水'], dtype='float32')

mixedData = pd.read_excel('混床.xlsx', header=0)
time2 = mixedData["运行时长"]
X2 = np.array(mixedData.iloc[:, 2:7], dtype='float32')
y2 = np.array(mixedData['出水'], dtype='float32')


doExpire = np.array([[14.00, 150, 150, 150, 150, 60]])
mixedExpire = np.array([[14.00, 1, 1, 1, 1, 0.6]])

if __name__ == "__main__":
    print(X1)
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