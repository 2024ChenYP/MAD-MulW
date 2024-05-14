class Config(object):
    def __init__(self):
        self.target_dataset = 'SleepEEG'
        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 3
        self.final_out_channels = 128

        self.num_classes = 5
        self.num_classes_target = 2
        self.dropout = 0.35
        self.features_len = 127
        self.features_len_f = 127

        self.increased_dim = 1
        # training configs
        self.num_epoch = 40

        self.TSlength_aligned = 178
        self.CNNoutput_channel = 10  # 90 # 10 for Epilepsy model

        # training configs
        self.num_epoch = 1


        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4 # 3e-4
        self.lr_f = self.lr

        # data parameters
        self.drop_last = True
        self.batch_size = 128
        """For Epilepsy, the target batchsize is 60"""
        self.target_batch_size = 128   # the size of target dataset (the # of samples used to fine-tune).

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

        self.num_channels = [150, 150, 150, 150]
        self.embedding_dim = 128
        self.seq_len = 1
        self.n_features = 3000


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