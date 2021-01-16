import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, values, slide_size):
        self.X = values
        self.X.dtype = 'float32'
        self.Y = values[slide_size:]
        self.Y.dtype = 'float32'
        self.window_size = slide_size

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, item):
        return self.X[item:item+self.window_size, :], self.Y[item]


class PredictNet(nn.Module):
    def __init__(self):
        super(PredictNet, self).__init__()
        self.lstm = nn.LSTM(3, 3, 4)
        self.fc = nn.Linear(3, 3)

    def forward(self, x_input):
        output, (x, cell) = self.lstm(x_input)
        x = x[-1, :, :]
        x = self.fc(x)
        x_out = F.relu(x)
        return x_out


def train():
    for epoch in range(epochs):
        for i, (feature, label) in enumerate(dl_train):
            feature = feature.view(8, bz, 3)
            pred_label = model(feature)
            loss = criterion(pred_label, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 1 == 0:
                print("epoch [{:>2}/{}], iter [{:>2}/{}], loss {:.4f}".
                      format(epoch, epochs, i, len(dl_train), loss.item()))


WINDOW_SIZE = 8
epochs = 500
bz = 84

if __name__ == '__main__':
    df = pd.read_csv("../data/covid-19/covid-19.csv", sep="\t")
    dfdata = df.set_index("date")
    dfdiff = dfdata.diff(periods=1).dropna()

    ds_train = MyDataset(dfdiff.values, WINDOW_SIZE)
    dl_train = DataLoader(ds_train, shuffle=True, batch_size=bz)

    model = PredictNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train()

