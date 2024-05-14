import argparse
from utils import *
from model1 import *
from train_model import TrainProcedure
from show import draw
import warnings
import time
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='seed value')
    parser.add_argument('--dataset', default='BGP', type=str, choices=['BGP', 'Epilepsy', 'KDD99', 'Coffee'])
    parser.add_argument('--idx', default=6, type=int, help='used BGP file index')   #1 5 6 10 14(13)(12)
    parser.add_argument('--device', default=0, type=int, help='used gpu number')
    return parser.parse_args()


if __name__ == "__main__":
    set_seed(0)
    args = get_args()
    device = torch.device(f"cuda:{args.device}")
    exec(f'from config_files.{args.dataset}_Configs import Config as Configs')
    configs = Configs()  # 由于目前只测试了2个数据集，只修改了2个configs

    # 加载实验数据，获取训练集与测试集
    if args.dataset == "BGP":
        dataset = torch.load(f'./data/{args.dataset}/cached_dataset_{args.idx}.pt')
    else:
        dataset = torch.load(f'./data/{args.dataset}/cached_dataset.pt')
    train_data, train_labels = dataset['train']['samples'], dataset['train']['labels']
    test_data, test_labels = dataset['test']['samples'], dataset['test']['labels']

    procedure = TrainProcedure(configs, device)

    # 开始训练
    print('====================Start Train====================')
    T1 = time.process_time()
    loss = []
    for epoch in range(1, configs.num_epoch + 1):
        print(f'Epoch:', epoch)
        loss.append(procedure.fit(train_data, train_labels))
    T2 = time.process_time()

    ax = plt.figure().gca()
    ax.plot(loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train'])
    plt.title('Loss over training epochs: GATwindow')
    plt.show()

    print('训练时间:%s毫秒' % ((T2 - T1) * 1000))
    print('训练时间:%s秒'% ((T2 - T1)))
    print('====================Start Test====================')
    draw(procedure.test(test_data, test_labels))

