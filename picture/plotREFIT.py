import numpy as np
import torch
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

output_dir = r"./"

df_train = pd.read_csv(f'../data/REFIT/FreezerRegularTrain_TRAIN.tsv', sep='\t', header=None)  # 读取完整包含标签的BGP数据集（地址）
df_test = pd.read_csv(f'../data/REFIT/FreezerRegularTrain_TEST.tsv', sep='\t', header=None)  # 读取完整包含标签的BGP数据集（地址）

df = pd.concat([df_train, df_test],axis=0)  # 纵向合并

# print(df.iloc[:,0])
nor = df[df.iloc[:,0] == 2]
abnor = df[df.iloc[:,0] == 1]

print(f'数据集大小：{df.shape}')
print('正常数据：%d，异常数据：%d'%(nor.shape[0], abnor.shape[0]))

labels = df.values[:, 0].astype(np.int64)
data = df.drop(columns=[0], axis=1).astype(np.float32)

# print(labels.shape, data.shape)

m, n = df.shape   # 3000 302
print(m, n)
x_axix = list(range(m))   # x轴为时序个数大小

fig = plt.figure(figsize=(11, 10))   # 设置画布大小
gs = GridSpec(11, 1, figure=fig)     # 划分画布
print('绘制所有时序，进行特征按列归一化，时序为横轴，归一化后结果为纵轴')

ax = fig.add_subplot(gs[0, 0])   # 首行
plt.title('Labels:REFIT')
# plt.plot(x_axix, [0 for _ in range(m)], label=f'character', alpha=0.7)
plt.plot(x_axix, labels, label=f'character', alpha=0.7)

ax = fig.add_subplot(gs[1:6, 0])
plt.xlabel('Time series')
plt.ylabel('The values of the features')
plt.title('Visualization of time series features:%Epilepsy without normalization')
for i in range(0, n-1):
    plt.plot(x_axix, data, label=f'character{i + 1}', alpha=0.7)

ax = fig.add_subplot(gs[6:11, 0])
plt.xlabel('Time series')
plt.ylabel('The values of the features')
plt.title('Visualization of time series features:Epilepsy with normalization')

data = pd.DataFrame(data=preprocessing.MinMaxScaler().fit_transform(data.T)).values  # 最大最小归一化
data = data.T
for i in range(0, n-1):
    plt.plot(range(len(data)), data[:, i], label=f'character{i + 1}', alpha=0.7)

plt.tight_layout()
plt.savefig('./ALL_time.jpg')
plt.close()