'''
    plot.py的作用
    输入：原始时间序列数据集
    输出：只包含正常数据的训练集，以及包含异常数据和正常数据混合的测试集
    步骤：1.划分数据集正常，异常
         2.正常部分划分
         3.合并得到训练集和测试集
'''
import os
import sklearn
import torch
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
i = 1 # 1 5 6 10 14
selection = True
if selection:
    if i == 0:
        data = 'dataset_code-red_513_1_rrc04'
    elif i== 1:
        data = 'dataset_code-red_559_1_rrc04'
    elif i== 2:
        data = 'dataset_code-red_6893_1_rrc04'

    elif i== 3:
        data = 'dataset_nimda_513_1_rrc04'
    elif i== 4:
        data = 'dataset_nimda_559_1_rrc04'
    elif i== 5:
        data = 'dataset_nimda_6893_1_rrc04'

    elif i== 6:
        data = 'dataset_slammer_513_1_rrc04'
    elif i== 7:
        data = 'dataset_slammer_559_1_rrc04'
    elif i== 8:
        data = 'dataset_slammer_6893_1_rrc04'

    elif i== 9:
        data = 'dataset_moscow_blackout_1853_1_rrc05'
    elif i== 10:
        data = 'dataset_moscow_blackout_12793_1_rrc05'
    elif i== 11:
        data = 'dataset_moscow_blackout_13237_rrc05'

    elif i== 12:
        data = 'dataset_malaysian-telecom_513_1_rrc04'
    elif i== 13:
        data = 'dataset_malaysian-telecom_20932_1_rrc04'
    elif i== 14:
        data = 'dataset_malaysian-telecom_25091_1_rrc04'
    elif i== 15:
        data = 'dataset_malaysian-telecom_34781_1_rrc04'

    elif i== 16:
        data = 'dataset_japan-earthquake_2497_1_rrc06'
    elif i== 17:
        data = 'dataset_japan-earthquake_10026_1_rv-sydney'
    # data = f'BGP/dataset_multi_code-red_513_1_rrc04.csv'              # 文件路径022
name = data
print('使用的数据集为：', data)

# 获取BGP数据集
df = pd.read_csv(f"../data/BGP/data_files/{data}.csv", sep=',', index_col=None)                    # 数据获取
f = df.drop(df.columns[0], axis=1)                                     # 去除第一列
print('数据集大小：',df.shape)                                            # 数据大小
# 划分正常异常数据
nor = df[df['class'] == 0]
abnor = df[df['class'] == 1]
print('正常数据：%d，异常数据：%d'%(nor.shape[0], abnor.shape[0]))

labels = f['class'].values
data = f.drop(['class', 'timestamp', 'timestamp2'], axis=1).values  # 去掉标签
# data = pd.DataFrame(data=preprocessing.MinMaxScaler().fit_transform(data)).values  # 最大最小归一化
# 获得标签和数据，利用数据绘制折线图
# 单一折线图绘制：单个特征
flag = 0
for i in range(len(labels)):
    if labels[i]^flag:    # 0：异常
        a = i             # 首次出现异常
        flag = labels[a]
        break
for i in range(a, len(labels)):
    if labels[i]^flag:
        b = i
        flag = labels[b]
        break
print(a, b)

x_axix = list(range(48))
# plt.subplot(2,2,1)
# plt.fill_between(range(a,b), [1 for _ in range(b-a)], color='red', alpha=1)
flag1 = 2
if flag1 == 0:
    print('绘制48个特征，每个特征为1个小图，时序作为横轴')
    plt.figure(figsize=(10, 12))
    for i in range(0,48):
        plt.subplot(12, 4, i+1)
        plt.grid(False)
        # plt.title('Data character')
        # plt.xlabel('times')
        # plt.ylabel(f'normalize{i+1}')
        plt.fill_between(range(24, 48), data[a:a+24, i], color='red', alpha=1)
        plt.plot(x_axix, data[a-24:a+24, i], label=f'character{i+1}',alpha=0.7)
    plt.tight_layout()
    plt.savefig('./character.pdf')
    plt.close()
    flag1 = 1

if flag1 == 1:
    plt.figure(figsize=(20, 15))
    print('绘制48个时序，每个时序为1个小图，特征为横轴')
    for i in range(0,48):
        plt.subplot(8, 6, i+1)
        plt.grid(False)
        # plt.title('Data character')
        # plt.xlabel('times')
        # plt.ylabel(f'normalize{i+1}')
        if i >= 24:
            plt.fill_between(range(48), data[a+i-24, :], color='red', alpha=1)
        plt.plot(x_axix, data[a+i-24, :], label=f'character{i+1}',alpha=0.7)
    plt.tight_layout()
    plt.savefig('./time.pdf')
    plt.close()

if flag1 == 2:
    fig = plt.figure(figsize=(11, 10))
    gs = GridSpec(11, 1, figure=fig)
    print('绘制所有时序，进行特征按列归一化，时序为横轴，归一化后结果为纵轴')

    ax = fig.add_subplot(gs[0, 0])
    plt.title('Labels:%s ' % name)
    plt.plot(range(len(data)), [0 for _ in range(len(data))], label=f'character{i + 1}', alpha=0.7)
    plt.fill_between(range(a, b), 0, 1, color='red', alpha=0.3)

    ax = fig.add_subplot(gs[1:6, 0])
    plt.xlabel('Time series')
    plt.ylabel('The values of the features')
    plt.title('Visualization of time series features:%s without normalization' % name)
    for i in range(0, 48):
        plt.plot(range(len(data)), data[:, i], label=f'character{i + 1}', alpha=0.7)

    ax = fig.add_subplot(gs[6:11, 0])
    plt.xlabel('Time series')
    plt.ylabel('The values of the features')
    plt.title('Visualization of time series features:%s with normalization' % name)
    # print(data.shape, data.T.shape)
    data = pd.DataFrame(data=preprocessing.MinMaxScaler().fit_transform(data)).values  # 最大最小归一化
    # for i in range(data.shape[1]):
    #     data[:, i] -= min(data[:, i])
    #     if max(data[:, i]) != 0:
    #         data[:, i] /= max(data[:, i])
    #     else:
    #         data[:, i] = data[:, i]

    for i in range(0, 48):
        plt.plot(range(len(data)), data[:, i], label=f'character{i + 1}', alpha=0.7)

    plt.tight_layout()
    plt.savefig('./ALL_time.jpg')
    plt.close()
print('--over--')