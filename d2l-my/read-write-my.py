import torch
from torch import nn
from torch.nn import functional as F


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, X):
        y = self.output(F.relu(self.hidden(X)))
        return y


class Test():
    X = torch.arange(4)
    y = torch.zeros(4)

    def test1(self):
        print(self.X)
        torch.save(self.X, 'X-file')
        X2 = torch.load('X-file')
        print(X2)

    def test2(self):
        print(self.X)
        print(self.y)
        # 保存张量列表
        torch.save([self.X, self.y], 'X-file')
        X3, y3 = torch.load('X-file')
        print(X3)
        print(y3)

    def test3(self):
        # 对 字典 的存储和读取
        mydict = {'X': self.X, 'y': self.y}
        print(mydict)
        torch.save(mydict, 'mydict')
        mydict2 = torch.load('mydict')
        print(mydict2)

    def test4(self):
        # 保存 模型参数 state_dict()
        net = Mlp()
        X4 = torch.rand(size=(2, 20))
        y = net(X4)

        # print(net)
        print(net.eval())
        # print(net.state_dict())
        print(y)

        torch.save(net.state_dict(), 'mlp.params')

        net_load = Mlp()
        net_load.load_state_dict(torch.load('mlp.params'))
        y2 = net_load(X4)

        # print(net_load)
        print(net_load.eval())
        # print(net_load.state_dict())
        print(y2)


if __name__ == '__main__':
    test = Test()
    # test.test1()
    # test.test2()
    # test.test3()
    test.test4()
