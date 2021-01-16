import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from netmodel.lenet5 import Lenet5
from netmodel.resnet import ResNet18
import torch.nn as nn
import torch.optim as optim

import os
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, file_path, transform):
        self.transform = transform
        class_list = os.listdir(file_path)
        self.label_container = []
        self.data_container = []
        for label, class_name in enumerate(class_list):
            class_dir = file_path + class_name + '/'
            for img_name in os.listdir(class_dir):
                img_dir = class_dir + img_name
                self.data_container.append(img_dir)
                self.label_container.append(label)

    def __getitem__(self, index):
        img = Image.open(self.data_container[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.label_container[index]
        return img, label

    def __len__(self):
        return len(self.label_container)


def main():
    # print(os.getcwd())
    train_path = '../data/cifar2/train/'
    test_path = '../data/cifar2/test/'

    batchsz = 32

    # cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor()
    # ]), download=True)

    cifar_train = MyDataset(train_path,
                            transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    # cifar_test = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor()
    # ]), download=True)

    cifar_test = MyDataset(test_path,
                           transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cpu')
    # model = Lenet5().to(device)
    model = ResNet18().to(device)

    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1000):
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('loss:', loss.item())

        # test
        model.eval()
        with torch.no_grad():
            total_num = 0
            total_correct = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num
            print("epoch:", epoch, "acc rate", acc)



if __name__ == '__main__':
    main()