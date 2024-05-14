import sys
sys.path.append("..")
from dataGAT import *
from model1 import AEModule
from dataloader import data_generator
from show import get_threshold


class TrainProcedure(nn.Module):
    def __init__(self, configs, device: torch.device):
        super(TrainProcedure, self).__init__()
        self.configs = configs
        self.device = device
        self.gat_module = GATModule(configs, device)
        self.ae_model = AEModule(configs).to(device)
        self.model_optimizer = torch.optim.Adam(self.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                                weight_decay=3e-4)

    def train_MulW_module(self, model: nn.Module, train_data):
        total_loss = []
        criterion = torch.nn.L1Loss(reduction='sum')
        for windows, data, labels in train_data:
            """数据传入原始数据及其加强数据以及标签"""
            FTCN, AE, score, flag, Zscore = model(windows, data)
            """Loss计算"""
            total_loss.append(0.05 * criterion(AE, data))

        return sum(total_loss) / len(total_loss)

    def fit(self, train_data, train_labels):
        att_train = torch.squeeze(self.gat_module(train_data))
        train_dict = {'samples': att_train, 'labels': train_labels}
        train_dl = data_generator(train_dict, self.configs, 1, subset=True)

        loss = self.train_MulW_module(self.ae_model, train_dl)
        print(loss)
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        return loss.item()

    def test(self, test_data, test_labels):
        self.ae_model.eval()
        att_test = torch.squeeze(self.gat_module(test_data))
        test_dict = {'samples': att_test, 'labels': test_labels}
        test_dl = data_generator(test_dict, self.configs, 1, subset=True)
        total_loss, losses, save_TCN, save_AE = [], list(), [], []
        init_data, pre_data = torch.empty(1, self.configs.n_features).to(self.device), torch.empty(1, self.configs.n_features).to(self.device)
        criterion = torch.nn.L1Loss(reduction='sum').to(self.device)
        with torch.no_grad():
            labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
            for j, (window, data, labels) in enumerate(test_dl, 0):
                FTCN, AE, score, TCNscore, Zscore = self.ae_model(window, data)
                if j == 0:
                    times2 = -torch.floor(torch.log10(criterion(AE[0], data.squeeze(1)[0])))
                """分类器计算"""
                for i in range(data.size(0)):
                    l_AEpre = criterion(AE[i], data.squeeze(1)[i])  # AE重构：重构误差
                    loss = (10 ** (times2 + 2)) * (5 * l_AEpre)  # 测试AE重构误差
                    losses.append(loss.cpu())
                    labels_numpy = labels[i].detach().cpu().numpy()  # 128*1
                    labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
                init_data = torch.vstack((init_data, data.squeeze(1)))
                pre_data = torch.vstack((pre_data, AE.squeeze(1)))
            labels_numpy_all = labels_numpy_all[1:]
            # """指标计算"""
            init, pre = init_data.cpu().detach().numpy(), pre_data.cpu().detach().numpy()
            init, pre = init[1:].T, pre[1:].T
            get_threshold(self.configs.target_dataset, labels_numpy_all, losses)
        return losses




