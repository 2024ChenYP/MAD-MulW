from torch.nn.utils import weight_norm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
device = torch.device("cuda:0")

# TCN的一些模块
class Chomp1d(nn.Module):            # 剪枝,一维卷积后会出现多余的padding
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        # 表示对继承自父类属性进行初始化
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        tensor.contiguous()会返回有连续内存的相同张量
        有些tensor并不是占用一整块内存，而是由不同的数据块组成
        tensor的view()操作依赖于内存是整块的，这时只需要执行
        contiguous()函数，就是把tensor变成在内存中连续分布的形式
        本函数主要是增加padding方式对卷积后的张量做切边而实现因果卷积
        """
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):             # 时序模块,两层一维卷积，两层Weight_Norm,两层Chomd1d，非线性激活函数为Relu,dropout为0.2
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))      # 权重归一化
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):        # 时序卷积模块,使用for循环对8层隐含层，每层25个节点进行构建。模型如下。 * 表示迭代器拆分layers为一层层网络
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数,输入层通道为1，隐含层是25。
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers = layers + [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)   # *作用是将输入迭代器拆成一个个元素

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)


# 编码器的一些模块
class Encoder(nn.Module):  # 编码器，包含双层LSTM
    """
    定义一个编码器的子类，继承父类 nn.Modul
    """

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features    # 1 48
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim   # 64 128
        # 使用双层LSTM
        self.rnn1 = nn.LSTM(
            input_size=n_features,   # 48
            hidden_size=self.hidden_dim,   # 128
            num_layers=1,
            batch_first=True)

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,   # 128
            hidden_size=n_features,    # 48
            num_layers=1,
            batch_first=True)

    def forward(self, x):
        # print(x.shape)
        # x = x.reshape((1, self.seq_len, self.n_features))
        # print(x.shape)
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.transpose(0,1)  #.reshape((1, self.embedding_dim))

class Encoder1(nn.Module):  # 编码器，包含双层LSTM
    """
    定义一个编码器的子类，继承父类 nn.Modul
    """

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder1, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features    # 1 48
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim   # 64 128
        # 使用双层LSTM
        self.linear1 = nn.Linear(1, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        # print('Encoder1', x.shape)
        # x = x.reshape((1, self.seq_len, self.n_features))
        # print('编码器使用单一线性层')
        x = self.linear1(x.transpose(1, 2))
        x = self.linear2(x)
        # x, (hidden_n, _) = self.rnn2(x)
        return x  #.reshape((1, self.embedding_dim))

class Encoder2(nn.Module):  # 编码器，包含双层LSTM
    """
    定义一个编码器的子类，继承父类 nn.Modul
    """

    def __init__(self, config, embedding_dim=64):
        super(Encoder2, self).__init__()

        self.seq_len, self.n_features, self.kernel_size = config.seq_len, config.n_features,config.kernel_size    # 1 48
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim   # 64 128
        # 使用双层LSTM
        self.cnn = nn.Conv2d(1, 4, 1, 1)
        self.pool = nn.MaxPool2d((1, self.kernel_size), stride=(1, self.kernel_size))

    def forward(self, x):
        x = self.cnn(x.unsqueeze(1))   # 128*1*1*48 --> 128*4*1*48
        x = self.pool(x)   # 128*4*1*48 --> 128*4*1*16
        return x  #.reshape((1, self.embedding_dim))


class Decoder(nn.Module):
    """
    定义一个解码器的子类，继承父类 nn.Modul
    """

    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=1,  # 64
            hidden_size=32,  # 64
            num_layers=1,
            batch_first=True)

        self.rnn2 = nn.LSTM(
            input_size=32,   # 64
            hidden_size=64,  # 128
            num_layers=1,
            batch_first=True)
        self.output_layer = nn.Linear(self.n_features, 1)   # 线性化

    def forward(self, x):
        # x = x.repeat(self.seq_len, 1)    # self.seq_len, self.n_features
        # x = x.reshape((1, self.seq_len, self.input_dim))    # self.n_features, self.seq_len, self.input_dim
        # x = x.transpose(0, 1)
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        # x = x.transpose(0, 1)
        # print(x.shape)
        # return self.output_layer(x).transpose(0, 1)
        return x

class Decoder1(nn.Module):
    """
    定义一个解码器的子类，继承父类 nn.Modul
    """

    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder1, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(16, 1, kernel_size=1)
        self.rnn1 = nn.RNN(input_size=1, hidden_size=32, num_layers=1)
        self.rnn2 = nn.RNN(input_size=32, hidden_size=64, num_layers=1)
        self.linear1 = nn.Linear(1, 32)
        self.linear2 = nn.Linear(32, 64)


        self.pooling = torch.nn.MaxPool2d(2)
        self.output_layer = nn.Linear(self.n_features, 1)   # 线性化
        self.fc = torch.nn.Linear(320, 10)
    def forward(self, x):
        # batch_size = x.size(0)
        # x = x.unsqueeze(1).transpose(2, 3)
        # x = F.relu((self.conv1(x)))
        # x = F.relu((self.conv2(x)))
        # x = x.view(batch_size, -1).unsqueeze(1)

        # print('解码器使用单一线性层')
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class Decoder2(nn.Module):  # 编码器，包含双层LSTM
    """
    定义一个编码器的子类，继承父类 nn.Modul
    """

    def __init__(self, config, embedding_dim=64):
        super(Decoder2, self).__init__()

        self.seq_len, self.n_features, self.kernel_size = config.seq_len, config.n_features,config.kernel_size    # 1 48
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim   # 64 128
        # 使用双层LSTM
        self.cnn = nn.Conv2d(4, 1, 1, 1)
        self.upsample = nn.Upsample((self.embedding_dim, self.n_features), mode='bilinear')

    def forward(self, x):
        x = self.cnn(x)   # 128*4*1*16 --> 128*1*1*16
        x = self.upsample(x).transpose(2,3)   # 128*1*1*16 --> 128*1*1*48   (128*fea*64)
        return x.squeeze(1)  #.reshape((1, self.embedding_dim))
# 并行

class AEModule(nn.Module):
    def __init__(self, configs):
        super(AEModule, self).__init__()
        self.tcn = TemporalConvNet(configs.n_features, num_channels=configs.num_channels, kernel_size=configs.kernel_size, dropout=configs.dropout)
        self.tcnlinear = nn.Linear(configs.num_channels[-1], configs.n_features)
        self.sig = nn.Sigmoid()

        self.zsize = configs.embedding_dim
        self.batch = configs.batch_size
        self.encoder = Encoder(configs.seq_len, configs.n_features, embedding_dim=configs.embedding_dim).to(device)
        self.decoder = Decoder(configs.seq_len, configs.embedding_dim, configs.n_features).to(device)
        self.encoder1 = Encoder1(configs.seq_len, configs.n_features, embedding_dim=configs.embedding_dim).to(device)
        self.decoder1 = Decoder1(configs.seq_len, configs.embedding_dim, configs.n_features).to(device)
        self.encoder2 = Encoder2(configs).to(device)
        self.decoder2 = Decoder2(configs).to(device)

        self.linear = torch.nn.Linear(configs.linearnum, 1)
        self.linear1 = torch.nn.Linear(configs.n_features, 1)
        self.linear2 = torch.nn.Linear(configs.embedding_dim, 1)

    def forward(self, windows, inputs):    # 128*10*48   128*1*48   ---> pre
        inputs = inputs.type(torch.FloatTensor).to(device)
        # print('input', inputs.device)

        """Lstm_AE由于内部是循环网络，每次输入为预测每一批中的单条时间序列，循环一组"""
        AEoutputs1 = inputs[0].unsqueeze(0).to(device)
        Zoutput = torch.zeros((1, self.zsize)).to(device)
        windows = windows.squeeze(2)
        # print('window', windows.shape)

        Z = self.encoder(windows).transpose(1, 2)   # encoder:LSTM
        Z1 = self.encoder1(inputs)  # encoder:linear
        Z2 = self.encoder2(inputs)  # encoder:CNN+pooling

        AEoutput = self.decoder(Z)#.transpose(1, 2)   # decoder:LSTM
        AEoutput1 = self.decoder1(Z1)    # decoder:linear
        AEoutput2 = self.decoder2(Z2)   # decoder:CNN+upsample


        flag = 1
        if flag == 1:
            """使用线性层还原"""
            AEoutputs = self.linear2(AEoutput).transpose(1, 2)
            TCNoutput = torch.zeros((1, self.zsize)).to(device)
        else:
            """F_TCN模型训练以每一批输入，整体计算"""
            inputs = inputs.transpose(1, 2)
            # print(inputs.shape, inputs.device, inputs.dtype)
            # print(inputs)
            TCNoutput = self.tcn(inputs) #.transpose(1, 2)  # 实际TCN出来的结果128*64*1  输入(Batch, input_channel, seq_len)
            """使用TCN还原：batch内每条时序都给定循环计算"""
            for i in range(self.batch):
                # print(AEoutput[i].shape, TCNoutput[i].shape)
                ad = torch.mm(AEoutput[i], TCNoutput[i]).unsqueeze(0).transpose(1, 2)
                AEoutputs1 = torch.cat((AEoutputs1, ad), 0)
            AEoutputs = AEoutputs1[torch.arange(AEoutputs1.size(0)) != 0]

        """AE内部获得z值，TCN最终获得结果进行拼接后得到值"""
        # score = torch.cat((Zoutput, TCNoutput1), 1)
        # scores = self.linear(score)
        scores = 1
        score1 = TCNoutput
        # score1 = self.linear1(TCNoutput1)
        score2 = self.linear2(Zoutput)
        # print(Z.shape)
        # print(AEoutputs1.shape)
        return self.sig(TCNoutput), self.sig(AEoutputs), scores, flag, score2


