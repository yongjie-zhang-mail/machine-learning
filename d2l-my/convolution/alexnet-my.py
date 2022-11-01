import torch
from d2l import torch as d2l
from torch import nn


class AlexNet:
    def __init__(self):
        super().__init__()

    def get_alexnet(self):
        # 处理 fashion-MNIST，入参通道数 和 出参通道数 和原论文 不太一样
        net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(11, 11), stride=(4, 4)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 卷积后全连接前，需要 flattern
            nn.Flatten(),
            # 上面 nn.MaxPool2d(kernel_size=3, stride=2) output shape: torch.Size([1, 256, 5, 5])
            # in_features=6400 计算逻辑：256*5*5=6400
            nn.Linear(in_features=6400, out_features=4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=10), nn.ReLU()
        )
        return net

    def test_alexnet_shape(self):
        # 批量大小、通道、高度、宽度
        X = torch.randn(size=(1, 1, 224, 224))
        alexNet = self.get_alexnet()
        for layer in alexNet:
            X = layer(X)
            print(f'{layer.__class__.__name__} output shape: {X.shape}')

    def train(self):
        batch_size = 128
        resize = 224
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=resize)

        net = self.get_alexnet()
        num_epochs = 10
        lr = 0.01
        # torch.device('cpu')
        device = d2l.try_gpu()
        d2l.train_ch6(net=net, train_iter=train_iter, test_iter=test_iter, num_epochs=num_epochs, lr=lr, device=device)


if __name__ == '__main__':
    alexNet = AlexNet()
    # alexNet.test1()
    alexNet.test_alexnet_shape()
