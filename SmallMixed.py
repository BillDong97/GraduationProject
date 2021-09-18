
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from LstmRnn import train
from data import x3, y3, time3, data_split


if __name__ == '__main__':
    # def data_split(x, y, time, train_data_ratio):
    #     data_len = len(x)
    #     train_len = int(train_data_ratio * data_len)
    #     x_train = x[train_len:]
    #     y_train = y[train_len:]
    #     x_test = x[:train_len]
    #     y_test = y[:train_len]
    #     t_test = time[:train_len]
    #
    #     return x_train, y_train, x_test, y_test, t_test
    # 划分数据集
    x_train, y_train, x_test, y_test, t_test = data_split(x3, y3, time3, 0.8)

    # 训练模型
    model = train(x_train, y_train, x_feature_num=8, y_feature_num=7, hidden_size=15, max_epochs=10000)

    # 测试模型
    model = model.eval()
    x_test_tensor = torch.from_numpy(x_test.reshape(1, -1, 8))
    y_predict_tensor = model(x_test_tensor)
    y_predict = y_predict_tensor.detach().numpy().reshape(-1, 7)

    # 作图
    plt.figure()
    plt.plot(t_test, y_test[:, -1], label='test-14')
    plt.plot(t_test, y_predict[:, -1], label='predict-14')
    plt.xlabel('operating time (h)')
    plt.ylabel('conductivity (us/cm)')
    plt.xlim(t_test[0], t_test[-1])
    plt.legend(loc='upper left')
    plt.savefig('./images/SmallMixedBed-14.jpeg')
    plt.show()
    plt.figure()
    plt.plot(t_test, y_test[:, -2], label='test-13')
    plt.plot(t_test, y_predict[:, -2], label='predict-13')
    plt.xlabel('operating time (h)')
    plt.ylabel('conductivity (us/cm)')
    plt.xlim(t_test[0], t_test[-1])
    plt.legend(loc='upper left')
    plt.savefig('./images/SmallMixedBed-13.jpeg')
    plt.show()
