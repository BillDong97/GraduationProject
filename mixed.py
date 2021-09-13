import numpy as np
from data import X2, y2, time2
import torch
from torch import nn
import matplotlib.pyplot as plt


class LstmRnn(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x


if __name__ == '__main__':
    data_len = len(X2)
    t = np.array(time2)

    # 划分训练集和测试集
    train_data_ratio = 0.8
    train_data_len = int(data_len * train_data_ratio)
    test_data_len = data_len - train_data_len
    X_train = X2[:train_data_len]
    y_train = y2[:train_data_len]
    INPUT_FEATURE_NUM = 5  # 输入特征值有五个：流量，进水，2L，3L，4L
    OUTPUT_FEATURE_NUM = 1  # 输出为出水
    t_train = t[:train_data_len]

    X_test = X2[train_data_len:]
    y_test = y2[train_data_len:]
    t_test = t[train_data_len:]

    # 将ndarray转化为tensor
    X_train_tensor = torch.from_numpy(X_train.reshape(1, train_data_len, INPUT_FEATURE_NUM))
    y_train_tensor = torch.from_numpy(y_train.reshape(1, train_data_len, OUTPUT_FEATURE_NUM))
    X_test_tensor = torch.from_numpy(X_test.reshape(1, test_data_len, INPUT_FEATURE_NUM))
    y_test_tensor = torch.from_numpy(y_test.reshape(1, test_data_len, OUTPUT_FEATURE_NUM))

    model = LstmRnn(INPUT_FEATURE_NUM, 500, output_size=OUTPUT_FEATURE_NUM)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    max_epochs = 20000
    for epoch in range(max_epochs):
        output = model(X_train_tensor)
        loss = loss_function(output, y_train_tensor)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if loss.item() < 1e-4:
            print('Epoch [{}/{}], loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            print('The loss value is reached')
            break
        elif (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))

    # 开始测试模型
    model = model.eval()

    y_predict = model(X_test_tensor)
    y_predict_plt = y_predict.detach().numpy().reshape(test_data_len)
    plt.figure()
    plt.plot(t_test, y_test, label='test')
    plt.plot(t_test, y_predict_plt, label='predict')
    plt.xlabel('t/h')
    plt.ylabel('y_predict and y_test')
    plt.xlim(t_test[0], t_test[-1])
    plt.ylim(0, 1.0)
    plt.show()
