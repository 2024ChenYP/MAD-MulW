import os
import sklearn
import torch
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import numpy as np
import random
from pyecharts import options as opts
from pyecharts.charts import HeatMap
from pyecharts.faker import Faker


def dtw(s1, s2):
    len_s1 = len(s1)
    len_s2 = len(s2)
    dp = [[float('inf')] * (len_s1 + 1) for _ in range(len_s2 + 1)]
    dp[0][0] = 0
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            # cost = (s1[i-1] - s2[j-1])*(s1[i-1] - s2[j-1])
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[-1][-1]

i = 6
lennum = 48
selection = True
if selection:
    if i == 0:
        data = 'dataset_code-red_513_1_rrc04'
    elif i== 1:
        data = 'dataset_code-red_559_1_rrc04'
    elif i== 2:
        data = 'dataset_code-red_6893_1_rrc04'

    elif i== 3:
        data = 'dataset_nimda_513_1_rrc04'
    elif i== 4:
        data = 'dataset_nimda_559_1_rrc04'
    elif i== 5:
        data = 'dataset_nimda_6893_1_rrc04'

    elif i== 6:
        data = 'dataset_slammer_513_1_rrc04'
    elif i== 7:
        data = 'dataset_slammer_559_1_rrc04'
    elif i== 8:
        data = 'dataset_slammer_6893_1_rrc04'

    elif i== 9:
        data = 'dataset_moscow_blackout_1853_1_rrc05'
    elif i== 10:
        data = 'dataset_moscow_blackout_12793_1_rrc05'
    elif i== 11:
        data = 'dataset_moscow_blackout_13237_rrc05'

    elif i== 12:
        data = 'dataset_malaysian-telecom_513_1_rrc04'
    elif i== 13:
        data = 'dataset_malaysian-telecom_20932_1_rrc04'
    elif i== 14:
        data = 'dataset_malaysian-telecom_25091_1_rrc04'
    elif i== 15:
        data = 'dataset_malaysian-telecom_34781_1_rrc04'

    elif i== 16:
        data = 'dataset_japan-earthquake_2497_1_rrc06'
    elif i== 17:
        data = 'dataset_japan-earthquake_10026_1_rv-sydney'
    # data = f'BGP/dataset_multi_code-red_513_1_rrc04.csv'              # 文件路径022
print('使用的数据集为：', data)
# 获取BGP数据集
df = pd.read_csv(f"../data/BGP/data_files/{data}.csv", sep=',', index_col=None)                    # 数据获取
f = df.drop(df.columns[0], axis=1)                                                                     # 数据大小
nor, abnor = df[df['class'] == 0], df[df['class'] == 1]
labels = f['class'].values
data = f.drop(['class', 'timestamp', 'timestamp2'], axis=1).values  # 去掉标签
print(f'未处理原始数据集大小：{df.shape}，正常数据：{nor.shape[0]}，异常数据：{abnor.shape[0]}')

# 获取异常开始点和结束点
flag = 0
for i in range(len(labels)):
    if labels[i]^flag:
        a = i
        flag = labels[a]
        break
for i in range(a, len(labels)):
    if labels[i]^flag:
        b = i
        flag = labels[b]
        break
print(f'异常开始点{a}, 异常结束点{b}')

Extractdata = data[a - int(lennum/2):a + int(lennum/2), :]
print(f'提取的数据集大小为{Extractdata.shape}，数据为{a - int(lennum/2)}-{a + int(lennum/2)}')

distance = []
maxdis = 0
for i in range(lennum):
    distance.append([])
# T_Extractdata = Extractdata.transpose()
for i in range(lennum):
    time1 = Extractdata[i]
    for j in range(lennum):
        time2 = Extractdata[j]
        distant = dtw(time1, time2)
        if distant>maxdis:
            maxdis = distant
        distance[i].append(int(distant))
# print(distance)

value = [[i, j, distance[i][j]] for i in range(lennum) for j in range(lennum)]
print(value)

# 一种热力图
c = (
    HeatMap()
    .add_xaxis(range(lennum))
    .add_yaxis("Time series", range(lennum), value)
    .set_global_opts(
        title_opts=opts.TitleOpts(title="HeatMap-Time"),
        visualmap_opts=opts.VisualMapOpts(max_=maxdis),
    )
    .render("heatmap_base.html")
)


# others
(
    HeatMap(init_opts=opts.InitOpts(width="1440px", height="720px"))
    .add_xaxis(xaxis_data=range(lennum))
    .add_yaxis(
        series_name="Punch Card",
        yaxis_data=range(lennum),
        value=value,
        label_opts=opts.LabelOpts(
            is_show=True, color="#fff", position="bottom", horizontal_align="50%"
        ),
    )
    .set_series_opts()
    .set_global_opts(
        legend_opts=opts.LegendOpts(is_show=False),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            splitarea_opts=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="category",
            splitarea_opts=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
        ),
        visualmap_opts=opts.VisualMapOpts(
            min_=0, max_=maxdis, is_calculable=True, orient="horizontal", pos_left="center"
        ),
    )
    .render("heatmap_on_cartesian.html")
)


# if __name__ == '__main__':
#     print('开始测试')



