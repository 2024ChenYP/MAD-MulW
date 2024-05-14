import numpy as np
import torch
from numpy import array
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
domain_down = 0
domain_up = 15845

test_data = torch.load("../data/KDD99/test.pt")
train_data = torch.load("../data/KDD99/train.pt")
output_dir = r"./"

trainsamples = train_data["samples"].squeeze(1).numpy()  # 样本
trainlabels = train_data["labels"].long().numpy()  # 标签
testsamples = test_data["samples"].squeeze(1).numpy()  # 样本
testlabels = test_data["labels"].long().numpy()  # 标签

sample = np.concatenate((trainsamples, testsamples), axis=0)
labels = np.concatenate((trainlabels, testlabels), axis=0)
labels = np.expand_dims(labels, axis=1)

print(sample.shape, labels.shape)
data = np.concatenate((sample, labels), axis=1)

m, n = data.shape  # m1=11500, n1=179
print(m, n)
nor = data[data[:, n - 1] == 1]
abnor = data[data[:, n - 1] == 0]
print('正常数据：%d，异常数据：%d'%(nor.shape[0], abnor.shape[0]))
# print(labels[0:100])

x_axix = list(range(m))   # x轴为时序个数大小

fig = plt.figure(figsize=(11, 10))   # 设置画布大小
gs = GridSpec(11, 1, figure=fig)     # 划分画布
print('绘制所有时序，进行特征按列归一化，时序为横轴，归一化后结果为纵轴')

ax = fig.add_subplot(gs[0, 0])   # 首行
plt.title('Labels:KDD99')
# plt.plot(x_axix, [0 for _ in range(m)], label=f'character', alpha=0.7)
plt.plot(x_axix[domain_down:domain_up], labels[domain_down:domain_up], label=f'character', alpha=0.7)
# abnornum = [i for i, x in enumerate(labels) if x == 0]
# print([i for i, x in enumerate(labels) if x == 1])
# plt.fill_between(abnornum[0:100], 0, 1, color='red', alpha=0.3)   # 这里不是ab了应该是所有1的下标的合集

ax = fig.add_subplot(gs[1:6, 0])
plt.xlabel('Time series')
plt.ylabel('The values of the features')
plt.title('Visualization of time series features:%Epilepsy without normalization')
for i in range(0, n-1):
    plt.plot(x_axix[domain_down:domain_up], data[domain_down:domain_up], label=f'character{i + 1}', alpha=0.7)

ax = fig.add_subplot(gs[6:11, 0])
plt.xlabel('Time series')
plt.ylabel('The values of the features')
plt.title('Visualization of time series features:Epilepsy with normalization')
print(data.shape, data.T.shape)

for i in range(n-1):   # 179
    data[:, i] -= min(data[:, i])
    if max(data[:, i]) != 0:
        data[:, i] /= max(data[:, i])
    else:
        data[:, i] = data[:, i]

for i in range(0, 5):
    plt.plot(range(len(data[domain_down:domain_up])), data[domain_down:domain_up, i], label=f'character{i + 1}', alpha=0.5)

plt.tight_layout()
plt.savefig('./ALL_time.jpg')
plt.close()