import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return self.corr2d(X, self.weight) + self.bias

    # 2d卷积
    def corr2d(self, X, K):
        h, w = K.shape
        Y = torch.zeros(size=(X.shape[0] - h + 1, X.shape[1] - w + 1))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
        return Y

    def test1(self):
        X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        Y = self.corr2d(X, K)
        print(X)
        print(K)
        print(Y)


if __name__ == '__main__':
    convLayer = ConvLayer()
    convLayer.test1()
