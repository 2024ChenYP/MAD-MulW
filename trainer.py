import sys
import matplotlib.pyplot as plt
import torch
sys.path.append("..")
from loss import * #NTXentLoss, NTXentLoss_poly
from show import *
from dataGAT import *
from model1 import AEModule

def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def Trainer(model, model_optimizer, train_dl, test_dl, device,
            logger, config, training_mode, model_F=None, model_F_optimizer=None,
            classifier=None, classifier_optimizer=None):
    """开始训练"""
    # logger.debug("Training started ....")
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')   # 学习率设置
    print('Train on Train datasts')
    performance_list = []
    """模型训练"""
    for epoch in range(1, config.num_epoch + 1):
        train_loss, init_data, flag, pre_data = model_finetune(model, train_dl, config, device, training_mode,model_optimizer, model_F=model_F,
                                    model_F_optimizer=model_F_optimizer,classifier=classifier, classifier_optimizer=classifier_optimizer)
        performance_list.append(train_loss)
        # logger.debug(f'\nEpoch : {epoch}\n'
        #              f'finetune Loss  : {train_loss:.4f}\t')
        # print('mean:', flag)

    # print(pre_data)
    # """绘制预测图"""
    # plt.plot(np.arange(0, len(init_data[0])), init_data[31], 'b*--', alpha=0.5, linewidth=1, label='init')
    # # plt.plot(np.arange(0, len(pre_data[0])), pre_data[31], 'r*--', alpha=0.5, linewidth=1, label='pre')
    # plt.ylabel('Loss')
    # plt.xlabel('time')
    # # plt.legend(['init', 'pre'])
    # plt.title('Loss over training epochs')
    # plt.savefig('PPP1.jpg')
    # plt.close()
    """loss显示"""
    lossnum = len(performance_list)
    ax = plt.figure().gca()
    ax.plot(performance_list)#[lossnum-30:lossnum])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train'])
    if flag == 1:
        plt.title('Loss over training epochs: Linear')
    else:
        plt.title('Loss over training epochs: GATwindow=', config.GATwindow)
    plt.savefig('loss.jpg')
    plt.close()
    saveloss = pd.DataFrame(columns=['score'], data=np.array(performance_list).T)
    saveloss.to_csv(f'./saveloss{flag}.csv')

    """模型测试"""
    # logger.debug('\nTest on Test datasts')
    a, b, init_data, pre_data = model_test(flag, model, test_dl, config, device)

    # plt.plot(np.arange(0, len(init_data[32])), init_data[31], 'b*--', alpha=0.5, linewidth=1, label='init')
    # plt.plot(np.arange(0, len(pre_data[32])), pre_data[31], 'r*--', alpha=0.5, linewidth=1, label='pre')
    # plt.ylabel('Loss')
    # plt.xlabel('time')
    # plt.legend(['init', 'pre'])
    # plt.title('Loss over training epochs')
    # plt.savefig('PPP2.jpg')
    # plt.close()
    # a, b, c, d = model_test(TCNmean, Zmean, mean, model, test_dl, config, device)
    # """预测差值显示"""
    # c = np.array(c).T
    # d = np.array(d).T
    # saveTCN = pd.DataFrame(columns=['TCN'], data=c)
    # saveAE = pd.DataFrame(columns=['AE'], data=d)
    # savescore = pd.DataFrame(columns=['score'], data=np.array(b).T)
    # saveAE.to_csv('./saveAE.csv')
    # saveTCN.to_csv('./saveTCN.csv')
    # savescore.to_csv('./savescore.csv')
    draw(b)
    # logger.debug("\n################## Training is Done! #########################")


def model_finetune(model, train_dl, config, device, training_mode, model_optimizer, model_F=None, model_F_optimizer=None,
                   classifier=None, classifier_optimizer=None):
    import time
    model.train()
    total_loss = []
    # pre_data = np.empty(shape=(128, 1, 48))
    # print(pre_data.shape)
    init_data, pre_data = torch.empty(1,config.n_features).to(device), torch.empty(1,config.n_features).to(device)
    criterion = torch.nn.L1Loss(reduction='sum').to(device)
    for windows, data, labels in train_dl:
        """数据传入原始数据及其加强数据以及标签"""
        print('data', data)
        window, data, labels = windows.float().to(device), data.float().to(device), labels.long().to(device)    # 传入data， labels
        """模型输出"""
        FTCN, AE, score, flag, Zscore = model(window, data)    # 输入数据以及其加强后的数据到FCN以及LstmAE，得到四个输出
        # TCNmean = torch.mean(TCNscore)
        # Zmean = torch.mean(TCNscore)
        # mean = torch.mean(TCNscore)
        """对比学习Loss计算"""
        l_AEpre = criterion(AE, data)   # 128*1*48  batch
        """Loss合计"""
        loss = 0.05 * l_AEpre  # 测试AE
        """分数计算"""
        # pre_data.append(AE.cpu().detach().numpy())
        init_data = torch.vstack((init_data, data.squeeze(1)))
        pre_data = torch.vstack((pre_data, AE.squeeze(1)))
        # print(pre_data.shape)
        total_loss.append(loss.item())
        """Loss回传更新模型"""
        ticks = time.process_time()
        model_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        model_optimizer.step()
        print("**********", time.process_time() - ticks)
    """指标计算"""
    init, pre = init_data.cpu().detach().numpy(), pre_data.cpu().detach().numpy()
    init, pre = init[1:].T, pre[1:].T
    # print(init.shape, pre)
    # print(data.shape)
    total_loss = torch.tensor(total_loss).mean()  # 取均值
    return total_loss, init, flag, pre  # 只返回loss结果


def model_test(Zmean, model, test_dl, config, device):    # 模型测试只用到了AE怎么用到TCN？以及如何更新score的权重参数
    model.eval()
    total_loss, losses, save_TCN, save_AE = [], list(), [], []
    init_data, pre_data = torch.empty(1,config.n_features).to(device), torch.empty(1,config.n_features).to(device)
    criterion = torch.nn.L1Loss(reduction='sum').to(device)
    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
        # print(Zmean)
        for j, (window, data, labels) in enumerate(test_dl, 0):

            """测试集数据输入加强的数据输入"""
            window, data, labels = window.float().to(device), data.float().to(device), labels.long().to(device)
            """模型输出"""
            FTCN, AE, score, TCNscore, Zscore = model(window, data)
            if j==0:
                times2 = -torch.floor(torch.log10(criterion(AE[0], data.squeeze(1)[0])))
            """分类器计算"""
            for i in range(data.size(0)):
                l_AEpre = criterion(AE[i], data.squeeze(1)[i])  # AE重构：重构误差
                loss = (10**(times2+2))*(5*l_AEpre)  # 测试AE重构误差
                losses.append(loss.cpu())

                labels_numpy = labels[i].detach().cpu().numpy()    # 128*1
                labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            init_data = torch.vstack((init_data, data.squeeze(1)))
            pre_data = torch.vstack((pre_data, AE.squeeze(1)))
    labels_numpy_all = labels_numpy_all[1:]
    # """指标计算"""
    init, pre = init_data.cpu().detach().numpy(), pre_data.cpu().detach().numpy()
    init, pre = init[1:].T, pre[1:].T
    get_threshold(config.target_dataset, labels_numpy_all, losses)
    return labels_numpy_all, losses, init, pre

