import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNN, self).__init__()
        # [100] => [256]
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=1)
        # [256*2] => [1]
        for p in self.rnn.parameters():
          nn.init.normal_(p, mean=0.0, std=0.001)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out, (hidden, cell) = self.rnn(x)
        # hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = hidden[-1]
        out = self.fc(hidden)
        return out

    def predict(self, x):
        out, (hidden, cell) = self.rnn(x)
        # hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = hidden[-1]
        out = self.fc(hidden)
        x = F.softmax(out)
        _, pred = torch.max(x, dim=1)
        return x, pred


class MyDataset(Dataset):
    def __init__(self, X, Y, slide_size):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.int64)
        self.window_size = slide_size

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, item):
        return self.X[item:item+self.window_size, :], self.Y[item]


def generate_data():
    # x = np.arange(0, 20 * np.pi, 1 * np.pi)
    # y = np.sin(x)
    # result = []
    # for i in range(x.shape[0] - WINDOW_SIZE):
    #     label = 1 if np.sum(y[i:i + WINDOW_SIZE]) > 0 else 0
    #     result.append(label)
    # y = result
    x = np.array([i for i in range(50)])
    y = np.array([i % 2 for i in range(49)])
    x = x.reshape(x.size, 1)
    return x, y


def train():
    avg_acc = []
    model.train()

    for i, (feature, label) in enumerate(dl_train):
        feature = feature.permute(1, 0, 2)
        pred_label = model(feature)
        # pred_label = pred_label.squeeze()
        loss = criterion(pred_label, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sf, preds = model.predict(feature)
        num_corrects = torch.eq(preds, label).sum().float().item()
        accs = num_corrects / feature.shape[1]

        avg_acc.append(accs)
        if i % 10 == 0:
            print("epoch [{:>2}/{}], iter [{:>2}/{}], loss {:.4f}, acc {:.4f}".
                  format(epoch, epochs, i, len(dl_train), loss.item(), accs))
            print(feature, label, preds, sf)
    avg_acc = np.array(avg_acc).mean()
    print("avg acc: ", avg_acc)


def evaluate():
    avg_acc = []

    model.eval()

    with torch.no_grad():
        for i, (feature, label) in enumerate(dl_train):
            feature = feature.permute(1, 0, 2)
            _, preds = model.predict(feature)
            num_corrects = torch.eq(preds, label).sum().float().item()
            accs = num_corrects / feature.shape[1]

            avg_acc.append(accs)

    avg_acc = np.array(avg_acc).mean()

    print('>>test:', avg_acc)


WINDOW_SIZE = 2
epochs = 1000
bz = 49


if __name__ == '__main__':
    x, y = generate_data()
    ds_train = MyDataset(x, y, WINDOW_SIZE)
    dl_train = DataLoader(ds_train, shuffle=False, batch_size=bz, drop_last=True)
    feature_dim = x.shape[1]
    model = RNN(feature_dim, 8)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        train()
        evaluate()
