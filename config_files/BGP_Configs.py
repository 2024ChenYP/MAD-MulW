import numpy as np

class Config(object):
    def __init__(self):
        self.target_dataset = 'BGP'
        # model configs
        self.input_channels = 1
        self.kernel_size = 3
        self.dropout = 0.35

        # training configs
        self.num_epoch = 10
        self.window = 11 # 15 AEwindow
        self.GATwindow = 15 # GATwindow

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1e-2 # 3e-4
        self.lr_f = self.lr

        # data parameters
        self.drop_last = True
        self.batch_size = 128 # 128
        """For Epilepsy, the target batchsize is 60"""
        self.target_batch_size = self.batch_size   # the size of target dataset (the # of samples used to fine-tune).

        self.num_channels = [128, 256, 128, 64]   # seq_len, num_channels, n_features
        self.embedding_dim = 64
        self.seq_len = 1  # 输入通道数-->1
        self.n_features = 48   # 序列长度--->27
        self.linearnum = self.n_features + self.embedding_dim
        self.thrange = np.arange(0, 20, 0.1)

