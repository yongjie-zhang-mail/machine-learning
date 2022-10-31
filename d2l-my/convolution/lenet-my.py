import torch
from d2l import torch as d2l
from torch import nn


class LeNet():
    def __init__(self):
        super().__init__()

    def test11(self):
        # it = iter([1,2,3,4,5])
        # y1 = next(it)
        # y2 = next(it)
        #
        # print(y1)
        # print(y2)
        #
        # y = next(iter([1, 2, 3, 4, 5]))
        # print(y)

        # 取了第一个元素？
        X = [1, None, None, 4, 5]
        device = None
        if not device:
            device = next(iter(X))

        print(device)

    def test12(self):
        # enumerate: 对 数组 加 下标索引
        e1 = enumerate([11, 22, 33, 44, 55])

        for i, X in e1:
            print(f'{i}: {X}')

    def get_lenet(self):
        net = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2), nn.Sigmoid(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)), nn.Sigmoid(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Flatten(),
                            nn.Linear(in_features=16 * 5 * 5, out_features=120), nn.Sigmoid(),
                            nn.Linear(in_features=120, out_features=84), nn.Sigmoid(),
                            nn.Linear(in_features=84, out_features=10))
        return net

    def evaluate_accuracy_gpu(self, net, data_iter, device=None):
        """
        计算 模型 在 数据集（测试集）上的精度；在哪种设备上执行（cpu,gpu）
        :param net:模型
        :param data_iter:测试集
        :param device:计算设备
        :return: accuracy:测试集精度
        """

        # 若 模型 是 nn.Module 的实例
        if isinstance(net, nn.Module):
            # 模型 设置为 评估模式
            net.eval()
            # 若 入参设备 为None，则取模型参数的 设备；取第一个参数的设备就行
            if not device:
                device = next(iter(net.parameters())).device

        # 累计字典：Key：正确预测的数量，总预测的数量
        metric = d2l.Accumulator(2)
        # 设置 不计算梯度
        with torch.no_grad():
            # 遍历 数据集
            for X, y in data_iter:
                # 若 X 为list的实例
                if isinstance(X, list):
                    # 数组的构造方式
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                # 计算 预测值 y_hat
                y_hat = net(X)
                # 计算预测正确的数量
                accurate_count = d2l.accuracy(y_hat, y)
                total_count = y.numel()
                metric.add(accurate_count, total_count)
        # 精度 = 正确预测的数量 / 总预测的数量
        accuracy = metric[0] / metric[1]
        return accuracy

    def train6_3(self, net, train_iter, test_iter, num_epochs, lr, device):
        def init_weights(m):
            m_type = type(m)
            if m_type == nn.Linear or m_type == nn.Conv2d:
                nn.init.xavier_normal_(m.weight)

        net.apply(init_weights)
        net.to(device)

        optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()
        num_batches = len(train_iter)

        timer = d2l.Timer()
        animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=('train loss', 'train acc', 'test acc'))

        for epoch in range(num_epochs):
            metric = d2l.Accumulator(3)
            net.train()
            for i, (X, y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()

                with torch.no_grad():
                    batch_total_count = X.shape[0]
                    batch_loss_count = batch_total_count * l
                    batch_acc_count = d2l.accuracy(y_hat, y)
                    metric.add(batch_loss_count, batch_acc_count, batch_total_count)

                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                if (i + 1) % (num_batches // 5) == 0 or (i + 1) == num_batches:
                    animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))

            test_acc = self.evaluate_accuracy_gpu(net, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))

        # animator.show()
        print(f'train loss:{train_l:.3f}, train acc:{train_acc:.3f}, test acc:{test_acc:.3f}')
        total_time = timer.sum()
        total_count = num_epochs * metric[2]
        print(f'speed: {total_count / total_time:.1f} examples/sec, on device {str(device)}')

    def test_lenet_shape(self):
        # 构建 LeNet 网络
        net = self.get_lenet()
        # 模拟 输入X
        X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
        # 打印每层输出的形状
        for layer in net:
            X = layer(X)
            print(f'{layer.__class__.__name__}, output shape: {X.shape}')

    def test_train_lenet(self):
        # 构建 LeNet 网络
        net = self.get_lenet()
        batch_size = 256
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
        lr, num_epochs = 0.9, 10
        self.train6_3(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


if __name__ == '__main__':
    leNet = LeNet()
    # leNet.test11()
    # leNet.test12()
    # leNet.test_lenet_shape()
    leNet.test_train_lenet()
