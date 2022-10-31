import torch
from torch import nn


class Pooling():
    def __init__(self):
        super().__init__()

    # 输入 X 的形状: nh, nw
    # 汇聚窗口 pool_size 的形状: ph, pw
    # 输出 Y 的形状: nh-ph+1, nw-pw+1
    def pool2d(self, X, pool_size, mode='max'):
        ph, pw = pool_size
        nh, nw = X.shape
        Y = torch.zeros(size=(nh - ph + 1, nw - pw + 1))
        yh, yw = Y.shape

        for i in range(yh):
            for j in range(yw):
                if mode == 'max':
                    Y[i, j] = X[i:i + ph, j:j + pw].max()
                elif mode == 'avg':
                    Y[i, j] = X[i:i + ph, j:j + pw].mean()

        return Y

    def test1(self):
        X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        Y1 = self.pool2d(X, (2, 2))
        Y2 = self.pool2d(X, (2, 2), 'avg')
        print(Y1)
        print(Y2)

    def test2(self):
        # 样本数=1，通道数=1
        X = torch.arange(end=16, dtype=torch.float32).reshape(1, 1, 4, 4)
        # 默认 stride步幅 = kernel_size窗口大小
        pool2d = nn.MaxPool2d(kernel_size=3)
        Y1 = pool2d(X)
        print(Y1)
        pool2d_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        Y2 = pool2d_2(X)
        print(Y2)
        # 自定义 步幅stride 和 填充padding
        pool2d_3 = nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3), padding=(0, 1))
        Y3 = pool2d_3(X)
        print(Y3)

    def test3(self):
        # 样本数=1，通道数=1
        X = torch.arange(end=16, dtype=torch.float32).reshape(1, 1, 4, 4)
        print(X)
        # 总维度不变
        XA = torch.cat((X, X + 1), dim=1)
        # 维度增加
        # XA = torch.stack((X,X+1),dim=1)
        print(XA)
        pool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        Y = pool2d(XA)
        print(Y)


if __name__ == '__main__':
    pooling = Pooling()
    # pooling.test1()
    # pooling.test2()
    pooling.test3()
