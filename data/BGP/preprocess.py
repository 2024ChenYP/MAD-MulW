import os
import torch
from sklearn import preprocessing
import pandas as pd

df = pd.read_csv("data_files/data.csv")             # 读取完整包含标签的BGP数据集（地址）
output_dir = r"./"                   # 保存路径

df = df.drop(df.columns[0], axis=1)                                     # 去除第一列
print('数据集大小：',df.shape)                                            # 数据大小

# 计算训练集和测试集包含的时间序列数量
train_num = round(df.shape[0] * 0.7)
test_num = df.shape[0]-train_num
# 划分正常异常数据
nor = df[df['class'] == 0]
abnor = df[df['class'] == 1]
print('正常数据：%d，异常数据：%d'%(nor.shape[0], abnor.shape[0]))

train_data = nor[:train_num]
test_data = pd.concat([nor[train_num:], abnor])

print('训练集数据大小：', train_data.shape[0])
print('测试集数据大小：', test_data.shape[0], '其中测试集包含正常时间序列 %d 个，异常时间序列 %d 个'%(test_data.shape[0]-abnor.shape[0],abnor.shape[0]))

train_labels = train_data['class']
train_data = train_data.drop(['class'], axis=1)  # 去掉标签
test_labels = test_data['class']
test_data = test_data.drop(['class'], axis=1)  # 去掉标签

train_data = pd.DataFrame(data=preprocessing.MinMaxScaler().fit_transform(train_data))  # 最大最小归一化
test_data = pd.DataFrame(data=preprocessing.MinMaxScaler().fit_transform(test_data))  # 最大最小归一化

dat_dict = dict()
dat_dict["samples"] = torch.Tensor(train_data.to_numpy()).unsqueeze(1)
dat_dict["labels"] = torch.from_numpy(train_labels.to_numpy())
torch.save(dat_dict, os.path.join(output_dir, "train.pt"))
print('训练集大小：',list(dat_dict.values())[0].shape)
#
# dat_dict = dict()
# dat_dict["samples"] = torch.from_numpy(X_val).unsqueeze(1)
# dat_dict["labels"] = torch.from_numpy(y_val)
# torch.save(dat_dict, os.path.join(output_dir, "val.pt"))
# print('验证集大小：',list(dat_dict.values())[0].shape)

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(test_data.to_numpy()).unsqueeze(1)
dat_dict["labels"] = torch.from_numpy(test_labels.to_numpy())
torch.save(dat_dict, os.path.join(output_dir, "test.pt"))
print('测试集大小：',list(dat_dict.values())[0].shape)

