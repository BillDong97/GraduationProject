
import torch
import matplotlib.pyplot as plt
from LstmRnn import train
from data import x1, y1, time1, data_split
from sklearn.metrics import r2_score


if __name__ == '__main__':
    # 划分数据集
    x_train, y_train, x_test, y_test, t_test = data_split(x1, y1, time1, 0.8)

    # 训练模型
    model = train(x_train, y_train, x_feature_num=5, y_feature_num=1, hidden_size=50, max_epochs=20000)

    # 测试模型
    model = model.eval()
    x_test_tensor = torch.from_numpy(x_test.reshape(1, -1, 5))
    y_predict_tensor = model(x_test_tensor)
    y_predict = y_predict_tensor.detach().numpy().reshape(-1)

    # 作图
    plt.figure()
    plt.plot(t_test, y_test, label='test')
    plt.plot(t_test, y_predict, label='predict')
    plt.xlabel('operating time (h)')
    plt.ylabel('DO (ppb)')
    plt.xlim(t_test[0], t_test[-1])
    plt.legend(loc='upper left')
    # plt.savefig('./images/doBed.jpeg')
    plt.show()

