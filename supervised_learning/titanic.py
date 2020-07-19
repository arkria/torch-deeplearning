import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime
import os


class DataSet:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item, :], self.Y[item]


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(15, 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = F.softmax(self.linear3(x), dim=1)
        x = self.linear3(x)  # the crossentropy loss will embed the softmax
        return x

    def predict(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=1)
        _, pred = torch.max(x, dim=1)
        return pred



def preprocessing(dfdata):
    dfresult = pd.DataFrame()

    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass' + str(x) for x in dfPclass.columns]
    dfresult = pd.concat([dfresult, dfPclass], axis=1)

    #Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)

    #age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    #SibSp, Parch, Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Cabin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return dfresult.values


def train():
    for epoch in range(epochs):
        lossTmp = []
        accTmp = []
        for i, (features, labels) in enumerate(trainDs):
            features = torch.tensor(features, dtype=torch.float32)
            onehot = model(features)
            preds = model.predict(features)
            losses = lossFunc(onehot, labels)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            num_corrects = torch.eq(preds, labels).sum().float().item()
            accs = num_corrects / features.shape[0]
            lossTmp.append(losses.item())
            accTmp.append(accs)
            if i % 1 == 0:
                print("epoch [{:>2}/{}], iter [{:>2}/{}], loss {:.4f}, accuracy {:.4f}".
                      format(epoch, epochs, i, len(trainDs), losses.item(), accs))
        writerTrain.add_scalar('loss', torch.tensor(lossTmp).mean().item(), epoch)
        writerTrain.add_scalar('accuracy', torch.tensor(accTmp).mean().item(), epoch)

        with torch.no_grad():
            feature = torch.tensor(dsValid.X, dtype=torch.float32)
            label = torch.tensor(dsValid.Y)
            onehot = model(feature)
            pred = model.predict(feature)
            loss = lossFunc(onehot, label)
            num_correct = torch.eq(pred, label).sum().float().item()
            acc = num_correct / feature.shape[0]
            print("=====> valid: epoch [{:>2}/{}], loss {:.4f}, accuracy {:.4f}".
                  format(epoch, epochs, loss.item(), acc))

            # for name, param in model.named_parameters():
            #     print(name, ": ", param)

    writerTrain.close()



bz = 64
lr = 0.01
epochs = 30


if __name__ == '__main__':
    dftrainRaw = pd.read_csv('../data/titanic/train.csv')
    dftestRaw = pd.read_csv('../data/titanic/test.csv')

    xTrain = preprocessing(dftrainRaw)
    xTest = preprocessing(dftestRaw)
    print("train data shape {}".format(xTrain.shape))
    print("test data shape {}".format(xTest.shape))

    yTrain = dftrainRaw['Survived'].values
    yTest = dftestRaw['Survived'].values

    currentTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    trainLogDir = 'logs/titanic/' + currentTime + '/train'
    validLogDir = 'logs/titanic/' + currentTime + '/valid'

    writerTrain = SummaryWriter(trainLogDir)
    writerValid = SummaryWriter(validLogDir)

    dsTrain = DataSet(xTrain, yTrain)
    trSize = int(0.95 * len(dsTrain))
    vlSize = len(dsTrain) - trSize
    dsTrain, dsValid = torch.utils.data.random_split(dsTrain, [trSize, vlSize])

    dsTrain = DataSet(dsTrain.dataset.X[dsTrain.indices,:], dsTrain.dataset.Y[dsTrain.indices])
    dsValid = DataSet(dsValid.dataset.X[dsValid.indices, :], dsValid.dataset.Y[dsValid.indices])
    dsTest = DataSet(xTest, yTest)

    trainDs = DataLoader(dsTrain, shuffle=False, batch_size=bz, drop_last=True)
    validDs = DataLoader(dsValid, shuffle=True, batch_size=bz, drop_last=True)

    testDs = DataLoader(dsTest, shuffle=True, batch_size=bz, drop_last=True)

    model = MLP()

    lossFunc = nn.CrossEntropyLoss()
    optimizer = optimizer.Adam(model.parameters(), lr=lr)

    train()

    model_path = './model/titanic/' + currentTime + '/train'
    os.makedirs(model_path)
    model_path = model_path + '/model.pkl'

    torch.save(model.state_dict(), model_path)

    model_dict = model.load_state_dict(torch.load(model_path))

    feature = torch.tensor(dsTest.X, dtype=torch.float32)
    label = torch.tensor(dsTest.Y)
    onehot = model(feature)
    pred = model.predict(feature)
    loss = lossFunc(onehot, label)
    num_correct = torch.eq(pred, label).sum().float().item()
    acc = num_correct / feature.shape[0]
    print("accuracy {:.4f}".format(acc))
