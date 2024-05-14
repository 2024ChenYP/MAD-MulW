import numpy as np

class Config(object):
    def __init__(self):
        self.target_dataset = 'Coffee'
        # model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 3   # base_model 用到了
        self.final_out_channels = 128

        self.num_classes = 2   # base_model 用到了
        self.num_classes_target = 2   # DNN分类 用到了
        self.dropout = 0.2
        self.features_len = 127   # base_model 用到了
        self.features_len_f = 127

        self.increased_dim = 1

        self.TSlength_aligned = 178  # 不知道有什么用，用的地方注释掉了
        self.CNNoutput_channel = 10  # 90 # 10 for Epilepsy model

        # training configs
        self.num_epoch = 100
        self.window = 4  # 4

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4 # 3e-4
        self.lr_f = self.lr

        # data parameters
        self.drop_last = True
        self.batch_size = 4
        """For Epilepsy, the target batchsize is 60"""
        self.target_batch_size = self.batch_size   # the size of target dataset (the # of samples used to fine-tune).

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

        self.num_channels = [128, 256, 128, 64]  # seq_len, num_channels, n_features
        self.embedding_dim = 64
        self.seq_len = 1  # 输入通道数-->1
        # if modeltest == 1:
        #     self.seq_len = self.batch_size
        # else:
        #     self.seq_len = self.target_batch_size
        self.n_features = 286   # 序列长度--->34
        self.linearnum = self.n_features+self.embedding_dim
        self.thrange = np.arange(0, 20, 0.1)


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.5
        self.jitter_ratio = 2
        self.max_seg = 12


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 64
        self.timesteps = 50