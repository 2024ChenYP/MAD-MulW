import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
# from augmentations import DataTransform_FD, DataTransform_TD, DataTransform


def create_data_seq(seq, time_window):
    window, pre = [], []
    l = len(seq)
    for i in range(l-time_window):
        x_tw = seq[i:i+time_window]
        y_tw = seq[i+time_window:i+time_window+1]
        window.append(x_tw)
        pre.append(y_tw)

    return window, pre


class Load_Dataset(Dataset):
    def __init__(self, dataset, config, training_mode, target_dataset_size=64, subset=False):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]    # 样本
        y_train = dataset["labels"]     # 标签

        """滑动"""
        self.window, self.pre = create_data_seq(X_train, config.window)
        self.labelwindow, self.labelpre = create_data_seq(y_train, config.window)
        # print(X_train.shape, self.pre.shape)

        # if len(X_train.shape) < 3:
        #     X_train = X_train.unsqueeze(2)
        #
        # if X_train.shape.index(min(X_train.shape)) != 1:  # 使Channels在第二个维度
        #     X_train = X_train.permute(0, 2, 1)
        #
        # if isinstance(X_train, np.ndarray):
        #     self.x_data = torch.from_numpy(X_train)
        #     self.y_data = torch.from_numpy(y_train).long()
        # else:
        #     self.x_data = X_train
        #     self.y_data = y_train

        # self.len = X_train.shape[0]
        # self.len = self.pre.shape[0]
        self.len = len(self.pre)

        """Augmentation"""
        # self.aug1, self.aug2 = DataTransform(self.x_data, config)   # 数据增强方式，采用的是另一篇论文中的强增强和弱增强

    def __getitem__(self, index):
        return self.window[index].squeeze(1), self.pre[index].squeeze(1), self.labelpre[index] #, self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]  # 两种增强方式

        # return self.x_data[index], self.y_data[index], self.aug1[index]  # 两种增强方式

    def __len__(self):
        return self.len


def data_generator(finetune_dataset, configs, training_mode, subset=True):

    finetune_dataset = Load_Dataset(finetune_dataset, configs, training_mode, target_dataset_size=configs.target_batch_size, subset=subset)

    """the valid and test loader would be finetuning set and test set."""
    train_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)
    return train_loader

if __name__ == '__main__':
    x = torch.load(os.path.join(f"./data/BGP", "train.pt"))