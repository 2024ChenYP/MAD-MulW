"""
dataGAT.py 对数据进行数据组操作和GAT操作
函数:GAT_pre()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv


def DataGroup(inputdata: torch.Tensor, window: int, mode: int = 1) -> torch.Tensor:
    """
    函数:data_group(data, window, mode)
    输入:原始数据表,窗口,模式
    操作:1.划分数据samples和labels
        2.原始数据向前增加n数目的第一条序列
        模式一:数据组包含当前时刻 n = window-1 用于GAT
        模式二:数据组不包含当前时刻 n = window 用于LSTM
        3.构建数据组，每一时刻由window个数据构成
    输出:数据组
    """
    # print('进行窗口建立，窗口：', window)
    # 检查输入数据的有效性
    if window < 1:
        raise ValueError("window必须大于等于1")
    if mode != 1 and mode != 2:
        raise ValueError("mode只能为1或2")

    samples = torch.Tensor(inputdata)  # 数据特征
    Addition = samples.cpu().numpy()
    FirstTime = np.expand_dims(Addition[0], axis=0)  # 获取第一条时序特征
    DataGroup = []
    if mode == 1:
        for w in range(window - 1):
            Addition = np.append(FirstTime, Addition, axis=0)  # 补全数据集
    elif mode == 2:
        for w in range(window):
            Addition = np.append(FirstTime, Addition, axis=0)  # 补全数据集
    for i in range(samples.shape[0]):
        DataGroup.append(Addition[i:i + window, :])

    # 返回数据组和标签
    return torch.Tensor(DataGroup)  # return 被测时序组


class GAT(nn.Module):
    """
    函数:GAT_pre()
    原理:注意力系数(公式),对自己和邻居的特征进行按照权重聚合
    输入:
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(GAT, self).__init__()
        self.layer1 = GATConv(input_dim, output_dim, num_heads=1)
        # self.layer2 = GATConv(hidden_dim, output_dim, num_heads=1)

    def forward(self, g: dgl.DGLGraph, feats: torch.Tensor):
        x = self.layer1(g, feats)
        # x = self.layer2(g, x)
        return x


class GATModule(nn.Module):
    def __init__(self, configs, device):
        super(GATModule, self).__init__()
        self.GAT = GAT(configs.n_features, 32, configs.n_features).to(device)
        self.window_size = configs.GATwindow
        self.device = device

    def data_division(self, ori_feats) -> torch.Tensor:
        return DataGroup(ori_feats, self.window_size)

    def graph_generate(self, num_nodes: int) -> dgl.DGLGraph:
        indices = ([i for i in range(num_nodes) for _ in range(num_nodes)],
                   [i for _ in range(num_nodes) for i in range(num_nodes)])
        return dgl.graph(indices)

    def forward(self, ori_feats):
        divided_feats = self.data_division(ori_feats).to(self.device)
        g = self.graph_generate(self.window_size).to(self.device)
        GAToutput = torch.stack([self.GAT(g, feats)[-1] for feats in divided_feats])
        return GAToutput


if __name__ == "__main__":
    num_nodes = 5
    print([[i for i in range(num_nodes) for _ in range(num_nodes)],
           [i for _ in range(num_nodes) for i in range(num_nodes)]])
    g = dgl.graph(([i for i in range(num_nodes) for _ in range(num_nodes)],
                   [i for _ in range(num_nodes) for i in range(num_nodes)]))
    print(g)
