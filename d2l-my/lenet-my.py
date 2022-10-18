import torch
from d2l import torch as d2l
from torch import nn


class LeNet():
    def __init__(self):
        super().__init__()

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
                total_count = y.numel
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

    def train6(self, net, train_iter, test_iter, num_epochs, lr, device):
        """
        训练函数 ver.6

        :param net:模型
        :param train_iter:训练集
        :param test_iter:测试集
        :param num_epochs:跑的次数
        :param lr:学习率
        :param device:计算的设备
        :return:
        """

        # 初始化 参数
        def init_weight(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_normal_(m.weight)

        net.apply(init_weight)

        # 模型 放在 gpu 上
        net.to(device)

        # 优化器：SGD(网络参数,学习率) 初始化
        optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)
        # 损失函数：交叉熵
        loss = nn.CrossEntropyLoss()

        # 画图
        # animator = d2l.Animator(xlabel='epoch', xlim=[1,num_epochs], legend=['train loss','train acc','test acc'])
        animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs], legend=['train loss', 'train acc', 'test acc'])
        # 定时器
        timer = d2l.Timer()
        # 训练集 的数据长度
        num_batches = len(train_iter)

        for epoch in range(num_epochs):

            metric = d2l.Accumulator(3)
            # 模型 设置为 训练模式
            net.train()
            # 遍历 训练集
            for i, (X, y) in enumerate(train_iter):
                timer.start()
                # 优化器 梯度 重置为0
                optimizer.zero_grad()
                # 数据集 放在 gpu 上
                X = X.to(device)
                y = y.to(device)
                # 预测 结果值 y_hat
                y_hat = net(X)
                # 计算 损失
                l = loss(y_hat, y)
                # 反向传播
                l.backward()
                # 做一次优化：计算梯度，更新 模型 参数
                optimizer.step()

                # 计算 模型 在测试集上的 精度
                with torch.no_grad():
                    # 预测正确的数量
                    accurate_count = d2l.accuracy(y_hat, y)
                    # X维度：样本数，通道，高度，宽度
                    total_count = X.shape[0]
                    # 训练损失的数量（训练损失之和），训练正确的数量（训练精确率之和），样本数
                    metric.add(l * total_count, accurate_count, total_count)

                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                # // 表示 除法 向下取整；
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))

            test_acc = self.evaluate_accuracy_gpu(net, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))
        # :.3f 保留3位小数的 float
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')

    def train6_2(self, net, train_iter, test_iter, num_epochs, lr, device):
        """
        训练函数 ver.6

        :param net: 模型
        :param train_iter: 训练集
        :param test_iter: 测试集
        :param num_epochs: 训练次数
        :param lr: 学习率
        :param device: 运算设备
        :return:
        """

        # 对 模型的 全链接层 和 卷积层，使用 xavier_normal_ 初始化 权重
        def init_weights(m):
            m_type = type(m)
            if m_type == nn.Linear or m_type == nn.Conv2d:
                nn.init.xavier_normal_(m.weight)

        net.apply(init_weights)
        # 模型 放在 device 上
        net.to(device)

        # 定义 优化器
        optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)
        # 定义 损失函数
        loss = nn.CrossEntropyLoss()

        # 定义 画图
        animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=('train loss', 'train acc', 'test acc',))
        # 定义 定时器
        timer = d2l.Timer()

        num_batches = len(train_iter)

        # 训练
        for epoch in range(num_epochs):
            # 定义 累积器
            metric = d2l.Accumulator(3)
            net.train()

            for i, (X, y) in enumerate(train_iter):
                timer.start()

                # 优化器 梯度 重置为0
                optimizer.zero_grad()

                # 训练集的X,y 放在 device 上
                X.to(device)
                y.to(device)
                # 计算 预测值
                y_hat = net(X)

                # 计算 损失
                l = loss(y_hat, y)
                # 反向传播
                l.backward()

                # 优化器 优化一次；计算梯度，并更新模型参数
                optimizer.step()

                timer.stop()

                # 评估 训练集 精度
                with torch.no_grad():
                    # X维度：样本数，通道数，高度，宽度
                    # 总数量
                    total_count = X.shape[0]
                    # 损失数量
                    loss_count = total_count * l
                    # 准确数量
                    accurate_count = d2l.accuracy(y_hat, y)

                    metric.add(total_count, loss_count, accurate_count)

                train_l = metric[1] / metric[0]
                train_acc = metric[2] / metric[0]

                if (i + 1) % (num_batches // 5) == 0 or (i + 1) == num_batches:
                    animator.add(epoch + ((i + 1) / num_batches), (train_l, train_acc, None))

            # 评估 测试集 精度
            test_acc = self.evaluate_accuracy_gpu(net, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))

        print(f'train_l:{train_l:.3f}, train_acc:{train_acc:.3f}, test_acc:{test_acc:.3f}')
        all_time = timer.sum()
        all_count = metric[0] * num_epochs
        print(f'{all_count / all_time:.1f} examples/sec on device {str(device)}')

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


if __name__ == '__main__':
    leNet = LeNet()
    leNet.test_lenet_shape()
    # leNet.test11()
    # leNet.test12()
