import torch
from torch import nn


class SimpleMlp(nn.Module):
    def __init__(self):
        super().__init__()

    def test(self):
        net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        X = torch.rand(size=(2, 4))
        y = net(X)

        print(X)
        print(y)


if __name__ == '__main__':
    # print(1)
    simpleMlp = SimpleMlp()
    simpleMlp.test()
