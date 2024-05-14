import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import heapq
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve
from sklearn import metrics

exec(f'from config_files.BGP_Configs import Config as Configs')
config = Configs() # THis is OK???


def draw(x):    # 主要是绘制模型输出值
    plt.plot(range(len(x)), x, 'b*--', alpha=0.5, linewidth=1, label='LOSS')
    plt.xlabel('time series')
    plt.ylabel('loss')
    plt.title('Test results for the time series')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('out.jpg')
    plt.close()

def histogram(name, f1, p, r, a, low, high, Delta):     # 绘制指标折线图
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(low, high, Delta), f1, 'b*--', alpha=0.5, linewidth=1, label='f1')
    plt.plot(np.arange(low, high, Delta), p, 'r*--', alpha=0.5, linewidth=1, label='precision')
    plt.plot(np.arange(low, high, Delta), r, 'y*--', alpha=0.5, linewidth=1, label='recall')
    plt.plot(np.arange(low, high, Delta), a, 'm*--', alpha=0.5, linewidth=1, label='accuracy')

    datasave = [f1, p, r, a]
    mid = pd.DataFrame(datasave)
    mid.to_csv('./datasave_csv.csv', header=False, index=False)

    plt.xlabel('thresholds')
    plt.ylabel('value')
    plt.title('The change of indexes under different thresholds:%s' % name)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # plt.savefig('thresholds.jpg')
    plt.show()
    plt.close()


def calculate(labels, pred, i):   # 指标计算
    a = np.round(accuracy_score(labels, pred), decimals=4)
    p = np.round(precision_score(labels, pred, average='weighted'), decimals=4)
    r = np.round(recall_score(labels, pred, average='weighted'), decimals=4)
    f1 = np.round(f1_score(labels, pred, average='weighted'), decimals=4)
    return f1, p, r, a


def get_threshold(name, labels, pred):    # 阈值滑动
    measure, f1, p, r, a = [], [], [], [], []
    if name == 'BGP' or name == 'KDD99':
        print('反转标签')
        for j in range(labels.size):   # 由于原始BGP数据标签正常数据为0，但是正样本在计算中标签值为1，所以进行转化（某些数据集不需要这个操作，手动注释）
            if labels[j] == 0:
                labels[j] = 1
            else:
                labels[j] = 0
    elif name == 'REFIT':
        print('改标签')
        for j in range(labels.size):   # 由于原始BGP数据标签正常数据为0，但是正样本在计算中标签值为1，所以进行转化（某些数据集不需要这个操作，手动注释）
            if labels[j] == 2:
                labels[j] = 1
            else:
                labels[j] = 0
    """返回阈值区间后的计算"""
    low, high, mea = adaptive_threshold(labels, pred)
    # if low < 0:
    #     low = 0
    # for i in config.thrange:   # 按照设定的阈值范围滑动
    Delta = (high - low) / 50
    # low = 0
    # high = int(max(pred).item())
    print('滑动范围：%d-%d,预测分数的均值为：%d' %(low, high, mea))
    for i in np.arange(low, high, Delta):  # 按照设定的阈值范围滑动
        # labels_1 = []
        # for j in range(len(pred)):
        #     if pred[j] <= i:      # 此处应该注意根据loss确定符号
        #         labels_1.append(1)
        #     else:
        #         labels_1.append(0)
        labels_1 = detection(pred, i)
        a1, b1, c1, d1 = calculate(labels, labels_1, i)
        f1.append(a1)
        p.append(b1)
        r.append(c1)
        a.append(d1)
    arr_max = heapq.nlargest(1, f1)  # 获取f1最大值
    index_max = map(f1.index, arr_max)  # 获取最大的值下标
    num = list(index_max)
    threshold = num[0]

    labels1 = detection(pred, low+threshold*Delta)
    savelabels = pd.DataFrame(columns=['labels'], data=np.array(labels1).T)
    savelabels.to_csv('./savelabels.csv')
    savelabels = pd.DataFrame(columns=['labels'], data=np.array(labels).T)
    savelabels.to_csv('./savelabelsinit.csv')
    classify_report_train = metrics.classification_report(labels, labels1)
    print('当前阈值选择为：', low+threshold*Delta)
    print(classify_report_train)
    print(p[threshold], r[threshold])
    print('阈值为%s时，准确率：%f, 精确率：%f，召回率：%f，此时获得最高的f1值：%f'%(round(np.arange(low, high, Delta)[threshold], 1), a[threshold], p[threshold], r[threshold], f1[threshold]))
    measure.append(a)
    measure.append(p)
    measure.append(r)
    measure.append(f1)
    save = np.array(measure).T
    df1 = pd.DataFrame(data=save, columns=['acc', 'pre', 'recall', 'f1'])
    df1.to_csv('./measure.csv')
    print(np.array(measure).T.shape)
    histogram(name, f1, p, r, a, low, high, Delta)


def adaptive_threshold(labels, pred):   # 动态阈值
    """
        1.得到输入的最小值，最大值，中间值和均值，
        2.如果中间值在均值上方，表示大部分数据位于中间值下方，计算一次中间值和均值的f1指标，以其中的大的一个作为新一轮的中间值计算区间？
        3.在最值和中间值中间计算新的中间值，比较新的中间值和均值大小
    """
    thre_max = int(max(pred).item())
    thre_min = int(min(pred).item())
    thre_mid = (thre_min+thre_max)/2
    thre_mean = int((sum(pred)/(len(pred))).item())

    """自定义方法"""
    # labels_1 = detection(pred, thre_mean)
    # f1_1 = f1_score(labels, np.array(labels_1))
    # labels_2 = detection(pred, thre_mid)
    # f1_2 = f1_score(labels, np.array(labels_2))
    # area = (thre_max-thre_mid)/thre_mean
    # thre_min = thre_mean-thre_mean
    thre_max = thre_mean+thre_mean+thre_mean+thre_mean

    # if thre_mean <= thre_mid:   # 第一轮选择，说明大部分数据位于中下位置
    #     if f1_1 <= f1_2:   # 均值处f1<=中间值f1
    #         low, high = thre_mean, thre_max
    #     else:
    #         low, high = thre_min, thre_min
    """
    二步均值法：
            初始选取T作为阈值，计算T两边的均值
            选取新的T=1/2两边均值和，重复操作
    结果：无效果
    """
    # T = thre_mean
    # for j in range(0, 3):
    #     mid1, mid2 = [], []
    #     for i in range(len(pred)):
    #         if pred[i] >= T:
    #             mid1.append(pred[i].item())
    #         else:
    #             mid2.append(pred[i].item())
    #     mid1, mid2 = np.mean(mid1), np.mean(mid2)
    #     T = (mid1+mid2)/2
    """
    确定离群值方法
        对数转换尝试
    """

    return thre_min, thre_max, thre_mean


def detection(pred, num):  # 输入预测值和阈值将预测值分割为正常和异常
    labels_1 = []
    for j in range(len(pred)):
        if pred[j] <= num:  # 此处应该注意根据loss确定符号
            labels_1.append(1)
        else:
            labels_1.append(0)
    return labels_1


def draw_ROC(labels, pred):     # 绘制ROC曲线
    auc = roc_auc_score(labels, pred)
    # auc = roc_auc_score(y_test,clf.decision_function(X_test))
    fpr, tpr, thresholds = roc_curve(labels, pred)
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('suhan.jpg', dpi=800)
    # plt.show()