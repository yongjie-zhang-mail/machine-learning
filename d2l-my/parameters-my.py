import torch
from torch import nn


class SimpleMlp(nn.Module):
    # 通用变量
    net: nn.Sequential = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.rand(size=(2, 4))
    y = net(X)

    # 构造函数
    def __init__(self):
        super().__init__()

    # 其它方法
    def block1(self):
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                             nn.Linear(8, 4), nn.ReLU())

    def block2(self):
        net = nn.Sequential()
        for i in range(4):
            net.add_module(f'block {i}', SimpleMlp.block1(self))
        return net

    def init_norm(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.zeros_(m.bias)

    def init_constant(self, m):
        if type(m) == nn.Linear:
            nn.init.constant_(m.weight, 1)
            nn.init.zeros_(m.bias)

    def init_xavier(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    def test1(self):
        print(self.X)
        print(self.y)

        print(self.net[2].state_dict())
        print(self.net[2].bias)
        print(self.net[2].bias.data)
        print(type(self.net[2].bias))
        print(self.net[2].weight.grad)

    def test2(self):
        print(*[(a, b) for a, b in self.net.named_parameters()])
        print(*[(a, b.shape) for a, b in self.net.named_parameters()])
        print(self.net.state_dict()['2.bias'].data)

    def test3(self):
        rgnet = nn.Sequential(SimpleMlp.block2(self), nn.Linear(4, 1))
        y2 = rgnet(self.X)
        print(y2)
        print(rgnet)
        print(rgnet[0][1][0].bias.data)

    def test4(self):
        self.net.apply(self.init_norm)
        # X形状 行: X.行, 列: net.in_features
        # 网络net 形状: in_features out_features
        print(self.net)
        # 网络参数值
        # weight 行: out_features, 列: in_features
        # bias 行: 1, 列: out_features
        print(self.net.state_dict())
        print(self.net[0].weight.data[0], self.net[0].bias.data[0])

    def test5(self):
        self.net.apply(self.init_constant)
        print(self.net)
        print(self.net.state_dict())

    def test6(self):
        # 不同块 使用 不同的初始化方法
        self.net[0].apply(self.init_xavier)
        # self.net[2].apply(self.init_norm)
        self.net[2].apply(self.init_constant)
        print(self.net)
        print(self.net.state_dict())

    def test7(self):
        # shared 为同一对象，更改会同步到所有引用到的地方
        shared = nn.Linear(8, 8)
        net7 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                             shared, nn.ReLU(),
                             shared, nn.ReLU(),
                             nn.Linear(8, 1))
        net7(self.X)
        print(net7)
        print(net7.state_dict())
        net7[2].weight.data[0][0] = 100
        print(net7.state_dict())


if __name__ == '__main__':
    # print(1)
    simpleMlp = SimpleMlp()
    # simpleMlp.test1()
    # simpleMlp.test2()
    # simpleMlp.test3()
    # simpleMlp.test4()
    # simpleMlp.test5()
    # simpleMlp.test6()
    simpleMlp.test7()
