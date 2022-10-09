import torch

from d2l import torch as d2l


class Channels():

    def __init__(self):
        super().__init__()

    def corr2d_multi_in(self, X, K):
        # zip 将对象中对应的元素打包成一个个元组
        # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起

        # 输入的通道数：ci
        # X的形状：ci * nh * nw
        # K的形状：ci * kh * kw
        return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

    def corr2d_multi_in_out(self, X, K):
        # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
        # 最后将所有结果都叠加在一起

        # 输出的通道数：co；输入的通道数：ci；
        # X的形状：ci * nh * nw
        # K的形状：co * ci * kh * kw
        # 小k的形状：ci * kh * kw
        return torch.stack([self.corr2d_multi_in(X, k) for k in K], dim=0)


    def test1(self):
        X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                          [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
        K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]],
                          [[1.0, 2.0], [3.0, 4.0]]])
        Y = self.corr2d_multi_in(X, K)
        print(Y)

    def test_stack(self):
        A = torch.tensor([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])

        B = torch.tensor([
            [11, 22, 33],
            [44, 55, 66],
            [77, 88, 99]
        ])

        # 根据不同维度进行拼接张量 tensor
        C0 = torch.stack((A, B), dim=0)
        C1 = torch.stack((A, B), dim=1)
        C2 = torch.stack((A, B), dim=2)

        print(C0)
        print(C1)
        print(C2)

    def test2(self):
        X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                          [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
        K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]],
                          [[1.0, 2.0], [3.0, 4.0]]])
        print(f'K shape {K.shape}')
        K2 = torch.stack((K, K + 1, K + 2), dim=0)
        print(f'K2 shape {K2.shape}')
        Y = self.corr2d_multi_in_out(X, K2)
        print(Y)

    # 输出的通道数：co；输入的通道数：ci；
    # X的形状：ci * nh * nw
    # K的形状：co * ci * kh * kw
    # Y的形状：co * nh2 * nw2
    def corr2d_multi_in_out_1X1(self, X, K):
        ci, nh, nw = X.shape
        co = K.shape[0]

        X = X.reshape((ci, nh * nw))
        K = K.reshape((co, ci))

        Y = torch.matmul(K, X)
        Y = Y.reshape((co, nh, nw))
        return Y

    # 输出的通道数：co；输入的通道数：ci；
    # X的形状：ci * nh * nw
    # K的形状：co * ci * kh * kw
    # Y的形状：co * nh2 * nw2
    def test3(self):
        X = torch.normal(mean=0, std=1, size=(3, 3, 3))
        K = torch.normal(mean=0, std=1, size=(2, 3, 1, 1))
        Y1 = self.corr2d_multi_in_out_1X1(X, K)
        Y2 = self.corr2d_multi_in_out(X, K)
        R = float(torch.abs(Y1 - Y2).sum()) < 1e-6
        print(Y1)
        print(Y2)
        print(R)
        assert R


if __name__ == '__main__':
    c = Channels()
    # c.test1()
    # c.test_stack()
    # c.test2()
    c.test3()
