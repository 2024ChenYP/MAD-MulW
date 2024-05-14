'''降维'''

# import numpy as np
# from sklearn.manifold import TSNE
#
# """将3维数据降维2维"""
#
# # 4个3维的数据
# x = np.array([[0, 0, 0, 1, 2], [0, 1, 1, 3, 5], [1, 0, 1, 7, 2], [1, 1, 1, 10, 22]])
# # 嵌入空间的维度为2，即将数据降维成2维
# ts = TSNE(n_components=2)
# # 训练模型
# ts.fit_transform(x)
# # 打印结果
# print(ts.embedding_)

'''对S型曲线数据的降维和可视化'''

# import matplotlib.pyplot as plt
# from sklearn import manifold, datasets
#
# """对S型曲线数据的降维和可视化"""
#
# # 生成1000个S型曲线数据
# x, color = datasets.make_s_curve(n_samples=1000, random_state=0)		# x是[1000,2]的2维数据，color是[1000,1]的一维数据
#
# n_neighbors = 10
# n_components = 2
#
# # 创建自定义图像
# fig = plt.figure(figsize=(8, 8))		# 指定图像的宽和高
# plt.suptitle("Dimensionality Reduction and Visualization of S-Curve Data ", fontsize=14)		# 自定义图像名称
#
# # 绘制S型曲线的3D图像
# ax = fig.add_subplot(211, projection='3d')		# 创建子图
# ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=color, cmap=plt.cm.Spectral)		# 绘制散点图，为不同标签的点赋予不同的颜色
# ax.set_title('Original S-Curve', fontsize=14)
# ax.view_init(4, -72)		# 初始化视角
#
# # t-SNE的降维与可视化
# ts = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
# # 训练模型
# y = ts.fit_transform(x)
# ax1 = fig.add_subplot(2, 1, 2)
# plt.scatter(y[:, 0], y[:, 1], c=color, cmap=plt.cm.Spectral)
# ax1.set_title('t-SNE Curve', fontsize=14)
# # 显示图像
# plt.show()

'''数据集的降维与可视化'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import torch
import sys
sys.path.append('../')
from data.BGP.BGPpre import preliminar
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# 加载数据
def get_data(dataname):
    """
	:return: 数据集、标签、样本数量、特征数量
	"""
    dataset = torch.load(os.path.join(f"../data/{dataname}", "total.pt"))
    data = dataset["samples"].squeeze(1).numpy()   # 样本
    label = dataset["labels"].numpy()   # 标签
    print(data.shape)
    # digits = datasets.load_digits(n_class=10)
    # data = digits.data  # 图片特征
    # print(data[1], data.shape)
    # label = digits.target  # 图片标签
    n_samples, n_features = data.shape  # 数据集的形状
    return data, label, n_samples, n_features


# 对样本进行预处理并画图
def plot_embedding(data, label, title):
    """
	:param data:数据集
	:param label:样本标签
	:param title:图像标题
	:return:图像
	"""
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    fig = plt.figure()  # 创建图形实例
    ax = plt.subplot(111)  # 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 5})
    plt.xticks()  # 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)
    # 返回值
    return fig


# 主函数，执行t-SNE降维
def plot_tSNE(dataname):
    data, label, n_samples, n_features = get_data(dataname)  # 调用函数，获取数据集信息
    print('Starting compute t-SNE Embedding...')
    ts = TSNE(n_components=2, init='pca', random_state=0)
    # t-SNE降维
    reslut = ts.fit_transform(data)
    print(data.shape, reslut.shape, "********")
    # 调用函数，绘制图像
    fig = plot_embedding(reslut, label, f't-SNE Embedding of {dataname}')
    # 显示图像
    plt.savefig('t-SNE.jpg')


def plot_resolve(dataname):
    data, label, n_samples, n_features = get_data(dataname)  # 调用函数，获取数据集信息
    print('Single variable time series: PCA...')
    ts = TSNE(n_components=1, init='pca', random_state=0)
    # pca降维
    reslut = ts.fit_transform(data)

    # 第一行observed：原始数据；第二行trend：分解出来的趋势部分；
    # 第三行seasonal：周期部分；最后residual：残差部分。
    period = 500

    decomposition = seasonal_decompose(data, period=period)  # (1)
    observe = decomposition.observed
    trend = decomposition.trend  # (2)
    cyclical = decomposition.seasonal  # (3)
    noise = decomposition.resid  # (4)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(range(len(reslut)), reslut, alpha=0.5, linewidth=1, label='Total')
    plt.xlabel('Time')
    plt.ylabel('value')
    plt.title(

    )
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(range(len(data)), observe, alpha=1, linewidth=1, label='observe')
    plt.plot(range(len(data)), trend, alpha=1, linewidth=1, label='trend')
    plt.plot(range(len(data)), cyclical, alpha=1, linewidth=1, label='cyclical')
    plt.plot(range(len(data)), noise, alpha=1, linewidth=1, label='noise')
    # plt.title('Combination')
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.subplot(4, 1, 1)
    plt.plot(range(len(data)), observe)#, 'b*--', alpha=0.5, linewidth=1, label='f1')
    plt.ylabel('observe')
    plt.subplot(4, 1, 2)
    plt.plot(range(len(data)), trend)#, 'r*--', alpha=0.5, linewidth=1, label='precision')
    plt.ylabel('trend')
    plt.subplot(4, 1, 3)
    plt.plot(range(len(data)), cyclical)#, 'y*--', alpha=0.5, linewidth=1, label='recall')
    plt.ylabel('cyclical')
    plt.subplot(4, 1, 4)
    plt.plot(range(len(data)), noise)#, 'm*--', alpha=0.5, linewidth=1, label='accuracy')
    plt.ylabel('noise')
    # decomposition.plot() #.suptitle('Time series are decomposed in terms of periods %s' % period)
    plt.suptitle('Time series are decomposed in terms of periods %s' % period)
    plt.tight_layout()
    plt.show()


# 主函数
if __name__ == '__main__':
    dataname = 'BGP'
    if dataname == 'BGP':
        preliminar(path='..')
    # plot_tSNE(dataname)
    # plt.figure(figsize=(10, 10))
    plot_resolve(dataname)