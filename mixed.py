
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from LstmRnn import train
from data import x2, y2, time2, data_split


if __name__ == '__main__':

    # 划分数据集
    x_train, y_train, x_test, y_test, t_test = data_split(x2, y2, time2, 0.5)

    # 训练模型
    model = train(x_train, y_train, 6, 1, 200, 20000)

    # 测试模型
    model = model.eval()
    x_test_tensor = torch.from_numpy(x_test.reshape(1, len(x_test), 6))
    y_predict_tensor = model(x_test_tensor)
    y_predict = y_predict_tensor.detach().numpy().reshape(len(y_test))
    print(r2_score(y_test, y_predict))

    # 作图
    plt.figure()
    plt.plot(t_test, y_test, label='test')
    plt.plot(t_test, y_predict, label='predict')
    plt.xlabel('operating time (h)')
    plt.ylabel('water (m3)')
    plt.xlim(t_test[0], t_test[-1])
    plt.legend(loc='upper left')
    # plt.savefig('./images/mixedBed.jpeg')
    plt.show()
