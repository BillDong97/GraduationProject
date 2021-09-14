
import matplotlib.pyplot as plt
from LstmRnn import train
from data import x1, y1, time1
from sklearn.metrics import r2_score


if __name__ == '__main__':

    # 训练模型
    model, t_test, x_test_tensor, y_test, test_data_len = train(x1, y1, time1, 20, 20000)

    # 开始测试模型
    model = model.eval()
    y_predict_tensor = model(x_test_tensor)

    # 作图
    y_predict = y_predict_tensor.detach().numpy().reshape(test_data_len)
    print(r2_score(y_test, y_predict))
    plt.figure()
    plt.plot(t_test, y_test, label='test')
    plt.plot(t_test, y_predict, label='predict')
    plt.xlabel('operating time (h)')
    plt.ylabel('DO (ppb)')
    plt.xlim(t_test[0], t_test[-1])
    plt.legend(loc='upper right')
    # plt.savefig('./images/doBed.jpeg')
    plt.show()
