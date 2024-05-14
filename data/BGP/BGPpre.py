import os

import numpy
import numpy as np
import sklearn
import torch
from sklearn import preprocessing
import pandas as pd

winsize = 15
# Weight = [0.9, 0.9, 0.9, 0.9, 0.9, 2, 0.9, 0.9, 0.8, 0.8, 0.8]
Weight = [1 for i in range(winsize)]
Weights = Weight/np.sum(Weight)

def win(data):
    print('进行数据窗口建立')
    print('窗口大小：', winsize)
    addition = data
    # print('***',addition.shape)  # 4665,48
    a = np.expand_dims(addition[0], axis=0)
    b = np.expand_dims(addition[-1], axis=0)

    # addition = np.append(
    #     np.transpose([Weights[:]]) * np.repeat([a], (winsize // 2)), addition, axis=0
    # )
    # addition = np.append(
    #     addition, np.transpose([Weights[::-1]]) * np.repeat([b], (winsize // 2)), axis=0
    # )

    for w in range(winsize//2):
        addition = np.append(a, addition, axis=0)
        addition = np.append(addition, b, axis=0)

    # a = np.append(a, a, axis=0)
    # b = np.append(b, b, axis=0)
    # addition = np.append(a, addition, axis=0)
    # addition = np.append(addition, b, axis=0)

    new = np.zeros(data.shape)
    print(new.shape, len(data))  # 4665,48  4665
    print(f'权重比例', Weights)
    for i in range(len(data)):
        for j in range(winsize):
            new[i] += Weights[j]*addition[i+j]
            # print('***', j, addition[i+j][0], new[i][0])
        # new[i] = new[i]/winsize
        # print(i, new[i][0])
    # print('***', new.shape)
    return new

def preliminar(path = '.'):
    i = 5 # 1 5 6 10 14(13)

    # for i in range(18):
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

    print('-------------------------------------------------------------------------')
    print('当前数据集为BGP：', data)
    df = pd.read_csv(f"{path}/data_files/{data}.csv")             # 读取完整包含标签的BGP数据集（地址）
    output_dir = f"{path}"                   # 保存路径

    df = df.drop(df.columns[0], axis=1)                                     # 去除第一列
    print('数据集大小：',df.shape)                                            # 数据大小
    nor = df[df['class'] == 0]
    abnor = df[df['class'] == 1]
    print('正常数据：%d，异常数据：%d'%(nor.shape[0], abnor.shape[0]))
    print(f'正负样本比例：{round(abnor.shape[0]/nor.shape[0], 4)}')

    # 计算训练集和测试集包含的时间序列数量
    train_num = round(nor.shape[0] * 0.7)
    test_num = df.shape[0]-train_num

    # train_data = nor[:int(test_num * 1.5)]
    train_data = nor[:train_num]
    test_data = pd.concat([nor[train_num:], abnor])
    # test_data = sklearn.utils.shuffle(test_data)
    # train_data = sklearn.utils.shuffle(train_data)

    print('训练集数据大小：', train_data.shape[0])
    print('测试集数据大小：', test_data.shape[0], '其中测试集包含正常时间序列 %d 个，异常时间序列 %d 个'%(test_data.shape[0]-abnor.shape[0],abnor.shape[0]))
    print(f'训练测试比例：{round(train_data.shape[0]/test_data.shape[0], 2)}')
    print(f'测试集正负样本比例：{round(abnor.shape[0]/(test_data.shape[0]-abnor.shape[0]), 2)}')

    total_labels = df['class']
    Total_data = df.drop(['class', 'timestamp', 'timestamp2'], axis=1).to_numpy()
    train_labels = train_data['class']
    Train_data = train_data.drop(['class', 'timestamp', 'timestamp2'], axis=1).to_numpy()  # 去掉标签
    test_labels = test_data['class']
    Test_data = test_data.drop(['class', 'timestamp', 'timestamp2'], axis=1).to_numpy()  # 去掉标签

    '''是否归一化'''
    # train_data = pd.DataFrame(data=preprocessing.MinMaxScaler().fit_transform(train_data))  # 最大最小归一化
    # test_data = pd.DataFrame(data=preprocessing.MinMaxScaler().fit_transform(test_data))  # 最大最小归一化
    '''是否窗口'''
    total_data = win(Total_data)
    train_data = win(Train_data)
    test_data = win(Test_data)

    dat_dict = dict()
    print(type(total_labels))
    dat_dict["samples"] = torch.Tensor(total_data)
    dat_dict["labels"] = torch.from_numpy(total_labels[:].to_numpy())
    torch.save(dat_dict, os.path.join(output_dir, "total.pt"))
    print('数据集集大小：', list(dat_dict.values())[0].shape)

    dat_dict = dict()
    dat_dict["samples"] = torch.Tensor(train_data)
    dat_dict["labels"] = torch.from_numpy(train_labels[:].to_numpy())
    torch.save(dat_dict, os.path.join(output_dir, "train.pt"))
    print('训练集大小：',list(dat_dict.values())[0].shape)

    # dat_dict = dict()
    # dat_dict["samples"] = torch.Tensor(train_data.to_numpy()).unsqueeze(1)
    # dat_dict["labels"] = torch.from_numpy(train_labels[30:].to_numpy())
    # torch.save(dat_dict, os.path.join(output_dir, "train.pt"))
    # print('训练集大小：',list(dat_dict.values())[0].shape)

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(test_data)
    dat_dict["labels"] = torch.from_numpy(test_labels[:].to_numpy())
    torch.save(dat_dict, os.path.join(output_dir, "test.pt"))
    print('测试集大小：',list(dat_dict.values())[0].shape)

    return Train_data, torch.from_numpy(train_labels[:].to_numpy()), Test_data, torch.from_numpy(test_labels[:].to_numpy())

preliminar('.')