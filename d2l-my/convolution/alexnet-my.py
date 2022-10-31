from torch import nn


class AlexNet:
    def __init__(self):
        super().__init__()

    def test1(self):
        print(1)

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
            # TODO: in_features=6400 这是怎么计算出来的？
            nn.Linear(in_features=6400, out_features=4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=10), nn.ReLU()
        )
        return net


if __name__ == '__main__':
    alexNet = AlexNet()
    alexNet.test1()
