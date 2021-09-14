import torch
from torch import nn


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


def train(x_train, y_train, x_feature_num, y_feature_num, hidden_size, max_epochs):
    train_data_len = len(x_train)
    # 将ndarray转化为tensor
    x_train_tensor = torch.from_numpy(x_train.reshape(1, train_data_len, x_feature_num))
    y_train_tensor = torch.from_numpy(y_train.reshape(1, train_data_len, y_feature_num))

    model = LstmRnn(x_feature_num, hidden_size, output_size=y_feature_num)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(max_epochs):
        output = model(x_train_tensor)
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
    return model
