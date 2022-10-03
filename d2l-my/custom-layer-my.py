import torch
from torch import nn
from torch.nn import functional as F


class CenteredLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X):
        X = X - X.mean()
        return X


class MyLinear(nn.Module):

    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(in_units, units))
        self.bias = nn.Parameter(torch.rand(units, ))

    def forward(self, X):
        y = F.relu(torch.matmul(X, self.weight.data) + self.bias.data)
        return y


class Test():
    net = CenteredLayer()

    def test1(self):
        X = torch.FloatTensor([1, 2, 3, 4, 5])
        y = self.net(X)
        print(X)
        print(y)

    def test2(self):
        X = torch.rand(4, 8)
        net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
        # 结果是 4行128列
        y = net(X)
        y_mean = y.mean()
        print(X)
        print(y)
        print(y_mean)

    def test3(self):
        net1 = MyLinear(5, 3)
        print(net1.weight)

        X1 = torch.rand(2, 5)
        y1 = net1(X1)

        print(X1)
        print(y1)

    def test4(self):
        net2 = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
        X2 = torch.rand(2, 64)
        y2 = net2(X2)

        print(X2)
        print(y2)


if __name__ == '__main__':
    test = Test()
    # test.test1()
    # test.test2()
    # test.test3()
    test.test4()
