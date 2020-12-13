import numpy as np
import torch
from torch import nn, optim
from matplotlib import pyplot as plt


num_time_steps = 50
pred_length = 1    # 通过pred_length来调控预测的长度
input_size = 1
hidden_size = 16
output_size = 1
lr = 0.01


class Net(nn.Module):

    def __init__(self, ):
        super(Net, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,    # true的时候，bz都到第一个dim了
        )
        for p in self.rnn.parameters():
          nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):

       out, hidden_prev = self.rnn(x, hidden_prev)
       # out, hidden_prev = self.rnn(x)
       # [b, seq, h]
       out = out.view(-1, hidden_size)
       out = self.linear(out)
       out = out.unsqueeze(dim=0)
       return out, hidden_prev


class LSTMNet(nn.Module):

    def __init__(self, ):
        super(LSTMNet, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,    # true的时候，bz都到第一个dim了
        )
        for p in self.lstm.parameters():
          nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev, cell_prev):

       # out, hidden_prev = self.rnn(x, hidden_prev)
       out, (hidden_prev, cell_prev) = self.lstm(x, [hidden_prev, cell_prev])
       # [b, seq, h]
       out = out.view(-1, hidden_size)
       out = self.linear(out)
       out = out.unsqueeze(dim=0)
       return out, hidden_prev


if __name__ == '__main__':
    model = LSTMNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    hidden_prev = torch.zeros(1, 1, hidden_size)    # [b, layer_num ,hidden_size]
    cell_prev = torch.zeros(1, 1, hidden_size)

    for iter in range(6000):
        start = np.random.randint(3, size=1)[0]
        time_steps = np.linspace(start, start + 10, num_time_steps)
        data = np.sin(time_steps)
        data = data.reshape(num_time_steps, 1)
        x = torch.tensor(data[:-pred_length]).float().view(1, num_time_steps - pred_length, 1)   # [bz, seq_len, feature_len]
        y = torch.tensor(data[pred_length:]).float().view(1, num_time_steps - pred_length, 1)

        output, hidden_prev = model(x, hidden_prev, cell_prev)
        hidden_prev = hidden_prev.detach()
        cell_prev = cell_prev.detach()

        loss = criterion(output, y)
        model.zero_grad()
        loss.backward()
        # for p in model.parameters():
        #     print(p.grad.norm())
        # torch.nn.utils.clip_grad_norm_(p, 10)
        optimizer.step()

        if iter % 100 == 0:
            print("Iteration: {} loss {}".format(iter, loss.item()))

    start = np.random.randint(3, size=1)[0]   #  随机初始化0,3之间的起始点
    time_steps = np.linspace(start, start + 10, num_time_steps)   # 将起始点之后的10长度内的离散为50个点
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)
    x = torch.tensor(data[:-pred_length]).float().view(1, num_time_steps - pred_length, 1)   # x_input是0-48
    y = torch.tensor(data[pred_length:]).float().view(1, num_time_steps - pred_length, 1)    # y_ouy 是1-49，目的是给0-48， 记住1-48，同时预测49

    predictions = []
    input = x[:, 0, :]
    for _ in range(x.shape[1]):
      input = input.view(1, 1, 1)
      (pred, hidden_prev) = model(input, hidden_prev, cell_prev)    # 这里的input不是一个sequence,而是只有一个点，当然也可以是sequence
      input = pred
      predictions.append(pred.detach().numpy().ravel()[0])

    x = x.data.numpy().ravel()
    y = y.data.numpy()
    plt.scatter(time_steps[:-pred_length], x.ravel(), s=90)
    plt.plot(time_steps[:-pred_length], x.ravel())

    plt.scatter(time_steps[pred_length:], predictions)
    plt.show()