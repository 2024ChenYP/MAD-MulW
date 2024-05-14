# import pandas as pd
# data = pd.read_csv('measure.csv')
# threshold = data.loc[:, ['threshold']].values.T.reshape(-1).tolist()
# acc = data.loc[:, ['acc']].values.T.reshape(-1).tolist()
# pre = data.loc[:, ['pre']].values.T.reshape(-1).tolist()
# recall = data.loc[:, ['recall']].values.T.reshape(-1).tolist()
# f1 = data.loc[:, ['f1']].values.T.reshape(-1).tolist()

from snapshot_selenium import snapshot
from pyecharts.render import make_snapshot

threshold = [0.18, 0.37, 0.55, 0.74, 0.92, 1.11, 1.29, 1.48, 1.66, 1.84]
acc = [78.76, 78.44, 74.97, 80.15, 87.68, 87.84, 88.92, 88.6, 88.36, 87.96]
pre = [48.6, 54.98, 53.78, 69.17, 81.91, 81.56, 83.09, 82.87, 82.34, 81.5]
recall = [49.96, 50.46, 51.9, 68.17, 79.85, 82.04, 83.91, 82.59, 82.79, 83.38]
f1 = [44.43, 46.05, 51.1, 68.86, 80.81, 81.8, 83.49, 82.73, 82.56, 82.37]

from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, Line

# x_data = ["{}月".format(i) for i in range(1, 13)]
bar = (
    Bar()
    .add_xaxis(threshold)
    .add_yaxis(
        "pre",
        # [2.0, 4.9, 7.0, 23.2, 25.6, 76.7, 135.6, 162.2, 32.6, 20.0, 6.4, 3.3],
        pre,
        yaxis_index=0,
        color="#70ad46",
        label_opts=opts.LabelOpts(is_show=False, font_size=20, font_family="Times New Roman"),
    )
    .add_yaxis(
        "recall",
        # [2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 175.6, 182.2, 48.7, 18.8, 6.0, 2.3],
        recall,
        yaxis_index=1,
        color="#4272c5",
        label_opts=opts.LabelOpts(is_show=False, font_size=20, font_family="Times New Roman"),
    )
    .extend_axis(
        yaxis=opts.AxisOpts(
            name="pre",
            type_="value",
            min_=40,
            max_=100,
            position="right",
            axisline_opts=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(color="#70ad46",),
            ),
            axislabel_opts=opts.LabelOpts(formatter="{value}%", font_size=20, font_family="Times New Roman"),
            name_textstyle_opts=opts.TextStyleOpts(font_size=20, font_family="Times New Roman")
        )
    )
    .extend_axis(
        yaxis=opts.AxisOpts(
            type_="value",
            name="f1 & acc",
            min_=0,
            max_=100,
            position="left",
            axisline_opts=opts.AxisLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color="#3E4A7B")),
            axislabel_opts=opts.LabelOpts(formatter="{value}%", font_size=20, font_family="Times New Roman"),
            splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=1)),
            name_textstyle_opts=opts.TextStyleOpts(font_size=20, font_family="Times New Roman")
        )
    )
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(type_="category", axislabel_opts=opts.LabelOpts(font_size=20, font_family="Times New Roman"), boundary_gap=True),
        yaxis_opts=opts.AxisOpts(name="recall", min_=40, max_=100, position="right", offset=50,
                                 axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color="#4272c5")),
                                 axislabel_opts=opts.LabelOpts(formatter="{value}%", font_size=20, font_family="Times New Roman"),
                                 name_textstyle_opts=opts.TextStyleOpts(font_size=20, font_family="Times New Roman")),
        title_opts=opts.TitleOpts(title="BGP-measure-train", pos_left="center", pos_top="top",
                                  title_textstyle_opts=opts.TextStyleOpts(font_size=30, font_family="Times New Roman"),),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        legend_opts=opts.LegendOpts(pos_top="8%", pos_left="60%",
                                    textstyle_opts=opts.TextStyleOpts(font_size=20, font_family="Times New Roman")),
    )
)

line1 = (
    Line()
    .add_xaxis(range(len(threshold)))
    .add_yaxis(
        "f1",
        # [2.0, 2.2, 3.3, 4.5, 6.3, 10.2, 20.3, 23.4, 23.0, 16.5, 12.0, 6.2],
        f1,
        yaxis_index=2,
        color="#675bba",
        label_opts=opts.LabelOpts(is_show=False),
    )
)

line2 = (
    Line()
    .add_xaxis(range(len(threshold)))
    .add_yaxis(
        "acc",
        # [2.0, 2.2, 3.3, 4.5, 6.3, 10.2, 20.3, 23.4, 23.0, 16.5, 12.0, 6.2],
        acc,
        yaxis_index=2,
        color="#5793f3",
        label_opts=opts.LabelOpts(is_show=False),
    )
)

bar.overlap(line1)
bar.overlap(line2)
grid = Grid(init_opts=opts.InitOpts(width="900px", height="500px"))
grid.add(bar, opts.GridOpts(pos_top="20%", pos_left="10%", pos_right="20%"), is_control_axis_index=True)
grid.render("BGP-measure-train-number.html")

# make_snapshot(snapshot, grid.render, "measure_train.jpg")