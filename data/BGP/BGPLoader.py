import os
import torch
import numpy as np
import pandas as pd
from default import _BGP_FILES_


def wrap_data(data: np.ndarray, labels: np.ndarray) -> dict:
    return {"samples": torch.from_numpy(data), "labels": torch.from_numpy(labels)}


def load_BGP_dataset(root_path: str = '.', file_idx: int = 0):
    print(_BGP_FILES_[file_idx])
    file = pd.read_csv(f"{root_path}/data_files/{_BGP_FILES_[file_idx]}.csv")
    file = file.drop(file.columns[0], axis=1)
    normal, abnormal = file[file['class'] == 0], file[file['class'] == 1]
    print('数据集大小：', file.shape)
    print('正常数据：%d，异常数据：%d' % (normal.shape[0], abnormal.shape[0]))
    print(f'正负样本比例：{round(abnormal.shape[0] / normal.shape[0], 4)}')

    # 计算训练集和测试集包含的时间序列数量
    train_num = round(normal.shape[0] * 0.7)
    train_data = normal[:int(train_num*0.4)]
    test_data = pd.concat([normal[train_num:], abnormal])
    print('训练集数据大小：', train_data.shape[0])
    print('测试集数据大小：', test_data.shape[0],
          '其中测试集包含正常时间序列 %d 个，异常时间序列 %d 个' % (
          test_data.shape[0] - abnormal.shape[0], abnormal.shape[0]))
    print(f'训练测试比例：{round(train_data.shape[0] / test_data.shape[0], 2)}')
    print(f'测试集正负样本比例：{round(abnormal.shape[0] / (test_data.shape[0] - abnormal.shape[0]), 2)}')

    total_labels = file['class'].to_numpy()
    total_data = file.drop(['class', 'timestamp', 'timestamp2'], axis=1).to_numpy()
    train_labels = train_data['class'].to_numpy()
    train_data = train_data.drop(['class', 'timestamp', 'timestamp2'], axis=1).to_numpy()  # 去掉标签
    test_labels = test_data['class'].to_numpy()
    test_data = test_data.drop(['class', 'timestamp', 'timestamp2'], axis=1).to_numpy()  # 去掉标签

    torch.save({'total': wrap_data(total_data, total_labels), 'train': wrap_data(train_data, train_labels),
                'test': wrap_data(test_data, test_labels)}, f"{root_path}/cached_dataset_{file_idx}.pt")


if __name__ == "__main__":
    # for i in range(18):
    #     load_BGP_dataset('.', i)
    load_BGP_dataset('.', 6)
