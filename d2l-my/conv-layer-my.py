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

    def test2(self):
        X = torch.ones(size=(6, 8))
        # print(X)
        X[:, 2:6] = 0
        print(X)

        K = torch.tensor([[1.0, -1.0]])
        Y = self.corr2d(X, K)
        print(Y)

        X2 = X.t()
        print(X2)
        Y2 = self.corr2d(X2, K)
        print(Y2)

    def test3(self):
        # 数据准备
        X = torch.ones(size=(6, 8))
        X[:, 2:6] = 0
        # print(X)
        K = torch.tensor([[1.0, -1.0]])
        Y = self.corr2d(X, K)
        # print(Y)
        # torch.nn 里面默认为4维张量：批量大小、通道、高度、宽度
        X = X.reshape(1, 1, 6, 8)
        Y = Y.reshape(1, 1, 6, 7)

        # 模型初始化
        conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), bias=False)
        lr = 3e-2
        # 跑10个epoch
        for i in range(10):
            # 求预测值
            Y_hat = conv2d(X)
            # 求损失（和实际值比较）
            l = (Y_hat - Y) ** 2
            # 梯度重置为0
            conv2d.zero_grad()
            # 求梯度：损失加和，反向传播
            l.sum().backward()
            # 权重的值 = 权重的值 - 学习率 * 权重的梯度（本次新计算的）
            # conv2d.weight.data[:] -= lr * conv2d.weight.grad
            conv2d.weight.data[:, :] -= lr * conv2d.weight.grad
            if (i + 1) % 2 == 0:
                print(f'epoch {i + 1}, loss {l.sum():.3f}')

        KX = conv2d.weight.data
        KX2 = KX.reshape(1, 2)
        print(KX)
        print(KX2)


if __name__ == '__main__':
    convLayer = ConvLayer(2)
    # convLayer.test1()
    # convLayer.test2()
    convLayer.test3()
