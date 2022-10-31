import torch
from d2l import torch as d2l
from torch import nn


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def mlp_concise_my():
    print("test print")
    # 构建模型
    # 输入: 打平 (28,28) -> (1,784)
    # 隐藏层1: 线性函数(784,256) + 激活函数(ReLU)
    # 输出: 线性函数(256,10)
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))
    # 初始化参数
    net.apply(init_weights)
    # 定义损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    # 定义训练函数(小批量随机梯度下降)
    lr = 0.1
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    # 获取数据集，分批次
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    # 训练模型
    num_epochs = 10
    d2l.train_ch3(net=net,
                  train_iter=train_iter,
                  test_iter=test_iter,
                  loss=loss,
                  num_epochs=num_epochs,
                  updater=trainer)


if __name__ == '__main__':
    mlp_concise_my()
