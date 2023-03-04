import torch
import torch.nn as nn


def naive_version():
    a = torch.eye(2)
    b = torch.tensor([-4, -2], dtype=torch.float32)
    x = torch.tensor([[1], [1]], dtype=torch.float32, requires_grad=True)

    for i in range(1000):
        y = torch.matmul(x.T, torch.matmul(a, x)) + torch.matmul(b, x)
        y.backward()
        print(f"iter {i}")
        print(f"y {y.item()}")
        print(f"x: ", x)
        print(f"x grad: ", x.grad)
        x = x - 0.001 * x.grad
        x = x.detach()  # detach from the origin graph
        x.requires_grad = True  # when detached, the requires_grad is False


class OptModel(nn.Module):
    def __init__(self):
        super(OptModel, self).__init__()
        self.x_opt = torch.nn.Parameter(torch.tensor([[0.5]], dtype=torch.float32))
        self.x_tmp = torch.tensor([[0.5]], dtype=torch.float32)

        self.a = torch.eye(2)
        self.b = torch.tensor([-4, -2], dtype=torch.float32)

    def get_val(self):
        self.x = torch.cat([self.x_opt, self.x_tmp], dim=0)
        return torch.matmul(self.x.T, torch.matmul(self.a, self.x)) + torch.matmul(self.b, self.x)


def OptModelVersion():
    opt_model = OptModel()
    optimizer = torch.optim.Adam(opt_model.parameters(), lr=0.01)

    for i in range(1000):
        print(f"==== iter {i}")
        y = opt_model.get_val()
        optimizer.zero_grad()
        y.backward()
        print(f"y {y.item()}")
        print(f"x: ", opt_model.x)
        print(f"x grad: ", opt_model.x.grad)
        optimizer.step()


if __name__ == '__main__':
    OptModelVersion()