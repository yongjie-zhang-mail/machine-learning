import math
import torch
from torch import nn


#@save
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # P 是 位置编码矩阵，形状为(1, max_len, num_hiddens)
        # 在初始化时已经计算好所有位置（0到max_len-1）的编码
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        # 输出 = 输入特征 + 位置特征
        
        # 第二个维度 :X.shape[1]：取前X.shape[1]个位置
        # X.shape[1]是当前输入序列的实际长度
        # 确保只使用与输入序列相同长度的位置编码
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


def test_P():
    """Test P."""
    max_len = 5
    num_hiddens = 6
    

    aa = torch.zeros((1, max_len, num_hiddens))
    print(aa)

    # 张量 切片； 
    # : 表示 改维度 完整保留； 
    # 第3个维度： 0 为 起始索引， :: 为 步长分隔符， 2 为 步长值
    aa[:, :, 0::2] = 2
    print(aa)

    # 张量 切片；0 为 起始索引， :: 为 步长分隔符， 2 为 步长值
    aa[:, :, 1::2] = 3
    print(aa)


def test_X():
    """Test X."""
    max_len = 5
    num_hiddens = 6

    # 位置矩阵：(max_len, 1)；pos, 生成序列，reshape 为 n 行 1 列 的张量；
    aa = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
    print(aa)

    # 生成 0 到 num_hiddens 的序列，步长为 2，为 1 行 d/2 列
    bb = torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
    # print(bb)

    # 频率矩阵：(1, num_hiddens//2)；生成 10000 的幂次方，指数为 张量；
    cc = torch.pow(10000, bb)
    print(cc)

    # 结果矩阵：(max_len, num_hiddens//2)；位置矩阵和频率矩阵相除，得到 X；广播机制自动扩展；
    X = aa / cc
    print(X)


def test_torch_arange():
    """Test torch.arange."""
    max_len = 5
    num_hiddens = 5

    aa = torch.arange(max_len, dtype=torch.float32)
    bb = aa.reshape(-1, 1)
    # print(aa)
    # print(bb)

    cc = torch.arange(0, num_hiddens, 2, dtype=torch.float32)
    # print(cc)

    dd = cc / num_hiddens
    print(dd)


def test_torch_pow():
    """Test torch.pow."""
    aa = torch.arange(0, 5, dtype=torch.float32)
    print(aa)
    
    bb = torch.pow(2, aa)
    print(bb)




if __name__ == "__main__":
    # print("hello")    
    # test_torch_arange()
    # test_torch_pow()
    # test_X()
    test_P()



