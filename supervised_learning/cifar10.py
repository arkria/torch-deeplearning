import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from netmodel.lenet5 import Lenet5
import torch.nn as nn
import torch.optim as optim

def main():
    batchsz = 32
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cpu')
    model = Lenet5().to(device)

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