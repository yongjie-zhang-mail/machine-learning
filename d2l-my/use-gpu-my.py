import torch


class Test():

    def try_gpu(self, i=0):
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda{i}')
        return torch.device('cpu')

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
        print(self.try_gpu())


if __name__ == '__main__':
    test = Test()
    # test.test1()
    test.test2()
