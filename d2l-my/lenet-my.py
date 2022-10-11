import torch
from torch import nn


class LeNet():
    def __init__(self):
        super().__init__()

    def test1(self):
        # 构建 LeNet 网络
        net = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2), nn.Sigmoid(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)), nn.Sigmoid(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Flatten(),
                            nn.Linear(in_features=16 * 5 * 5, out_features=120), nn.Sigmoid(),
                            nn.Linear(in_features=120, out_features=84), nn.Sigmoid(),
                            nn.Linear(in_features=84, out_features=10))
        # 模拟 输入X
        X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
        # 打印每层输出的形状
        for layer in net:
            X = layer(X)
            print(f'{layer.__class__.__name__}, output shape: {X.shape}')


if __name__ == '__main__':
    leNet = LeNet()
    leNet.test1()
