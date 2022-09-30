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


if __name__ == '__main__':
    # 随机初始化 张量 2行20列
    X = torch.rand(2, 20)
    net = MLP()
    print(X)
    print(net(X))
