import torch
from torch import nn
from torch.nn import functional as F


# 使用 torch.nn.Module 定义 MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for index, module in enumerate(args):
            self._modules[str(index)] = module

    def forward(self, X):
        for module in self._modules.values():
            X = module(X)
        return X


if __name__ == '__main__':
    # 随机初始化 张量 2行20列
    X = torch.rand(2, 20)

    # 模型构造
    # net = MLP()
    net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    # 前向传播
    y = net(X)

    # 打印
    print(X)
    print(y)
