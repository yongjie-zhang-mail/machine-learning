import torch
from torch import nn

class Test():

    def try_gpu(self, i=0):
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')

    def try_all_gpus(self):
        gpus = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        return gpus if gpus else [torch.device('cpu')]

    def test1(self):
        d1 = torch.device('cpu')
        d2 = torch.device('cuda')
        d3 = torch.device('cuda:1')

        gpu_count = torch.cuda.device_count()

        print(d1)
        print(d2)
        print(d3)
        print(gpu_count)

    def test2(self):
        y1 = self.try_gpu()
        print(y1)
        y2 = self.try_gpu(10)
        print(y2)
        y3 = self.try_all_gpus()
        print(y3)

    def test3(self):
        X = torch.tensor([1, 2, 3])
        print(X.device)

        X2 = torch.ones(size=(2, 3), device=self.try_gpu())
        print(X2)

    def test4(self):
        net = nn.Sequential(nn.Linear(3, 1))
        net = net.to(device=self.try_gpu())
        X = torch.ones(size=(2, 3), device=self.try_gpu())
        y = net(X)
        print(y)
        print(net)
        print(net[0].weight)


if __name__ == '__main__':
    test = Test()
    # test.test1()
    # test.test2()
    # test.test3()
    test.test4()
