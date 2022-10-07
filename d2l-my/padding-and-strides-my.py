import torch
from torch import nn


class PaddingAndStrides():

    def compConv2d(self, conv2d, X):
        # 2维转4维，使用pytorch做运算，运算完，再转换回去
        # torch.nn 里面默认为4维张量：批量大小、通道、高度、宽度
        X = X.reshape((1, 1) + X.shape)
        Y = conv2d(X)
        Y = Y.reshape(Y.shape[2:])
        return Y

    def test1(self):
        # 𝑝ℎ=𝑘ℎ−1和𝑝𝑤=𝑘𝑤−1
        conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1)
        X = torch.rand(size=(8, 8))
        Y = self.compConv2d(conv2d, X)
        print(Y.shape)

    def test2(self):
        # ph = (kh-1)/2, pw = (pw-1)/2;
        # ph = (5-1)/2, pw = (3-1)/2;
        conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
        X = torch.rand(size=(8, 8))
        Y = self.compConv2d(conv2d, X)
        print(Y.shape)

    def test3(self):
        # ⌊(𝑛ℎ−𝑘ℎ+𝑝ℎ+𝑠ℎ)/𝑠ℎ⌋×⌊(𝑛𝑤−𝑘𝑤+𝑝𝑤+𝑠𝑤)/𝑠𝑤⌋.
        # 输出形状将为(𝑛ℎ/𝑠ℎ)×(𝑛𝑤/𝑠𝑤)
        conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1, stride=(2, 2))
        X = torch.rand(size=(8, 8))
        Y = self.compConv2d(conv2d, X)
        print(Y.shape)

    def test4(self):
        conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
        X = torch.rand(size=(8, 8))
        Y = self.compConv2d(conv2d, X)
        print(Y.shape)


if __name__ == '__main__':
    instance = PaddingAndStrides()
    # instance.test1()
    # instance.test2()
    # instance.test3()
    instance.test4()
