import pandas as pd
data = pd.read_csv('measure.csv')
# threshold = data.loc[:, ['threshold']].values.T.reshape(-1).tolist()
acc = data.loc[:, ['acc']].values.T.reshape(-1).tolist()
pre = data.loc[:, ['pre']].values.T.reshape(-1).tolist()
recall = data.loc[:, ['recall']].values.T.reshape(-1).tolist()
f1 = data.loc[:, ['f1']].values.T.reshape(-1).tolist()
data = pd.read_csv('measure.csv', dtype=str)
threshold = data.loc[:, ['threshold']].values.T.reshape(-1).tolist()

from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, Line
from snapshot_selenium import snapshot
from pyecharts.render import make_snapshot

bar = (
    Line()
    .add_xaxis(threshold)
    .add_yaxis("acc", acc, color="#70ad46", label_opts=opts.LabelOpts(is_show=False, font_size=20, font_family="Times New Roman"),
          markpoint_opts=opts.MarkPointOpts(
              data=[opts.MarkPointItem(type_="max", name="最大值"),],
              symbol_size=[60,60],

          ),
               )
    .add_yaxis("pre", pre, color="#4272c5", label_opts=opts.LabelOpts(is_show=False, font_size=20, font_family="Times New Roman"),
          markpoint_opts=opts.MarkPointOpts(
              data=[opts.MarkPointItem(type_="max", name="最大值"),],
              symbol_size=[60,60],
          )
               )
    .add_yaxis("recall", recall, color="#3E4A7B", label_opts=opts.LabelOpts(is_show=False, font_size=20, font_family="Times New Roman"),
          markpoint_opts=opts.MarkPointOpts(
              data=[opts.MarkPointItem(type_="max", name="最大值")],
              symbol_size=[60,60],
          )
               )
    .set_series_opts(
        areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
        label_opts=opts.LabelOpts(is_show=False),)
    .set_global_opts(title_opts=opts.TitleOpts(title="acc & pre & recall & f1", pos_left="center", pos_top="8%",
                                  title_textstyle_opts=opts.TextStyleOpts(font_size=30, font_family="Times New Roman"),),
                     xaxis_opts=opts.AxisOpts(type_="category",
                                              axislabel_opts=opts.LabelOpts(font_size=20, font_family="Times New Roman"), boundary_gap=True),
                     yaxis_opts=opts.AxisOpts(name="acc & pre & recall", min_=40, max_=100, position="left",
                                              # axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color="#4272c5")),
                                              axislabel_opts=opts.LabelOpts(formatter="{value}%", font_size=20, font_family="Times New Roman"),
                                              name_textstyle_opts=opts.TextStyleOpts(font_size=20, font_family="Times New Roman")),
                     legend_opts=opts.LegendOpts(pos_top="13%", pos_right="10%",
                                                 textstyle_opts=opts.TextStyleOpts(font_size=20, font_family="Times New Roman")),
                     )

)
line = (
    Line()
    .add_xaxis(threshold)
    .add_yaxis("f1", f1, label_opts=opts.LabelOpts(is_show=False, font_size=20, font_family="Times New Roman"),
               markpoint_opts=opts.MarkPointOpts(
                   data=[opts.MarkPointItem(type_="max", name="最大值"),],
                   symbol_size=[60,60],
               ),
               linestyle_opts=opts.LineStyleOpts(color="#26547C", width=2),
               itemstyle_opts=opts.ItemStyleOpts(border_width=3, color="#26547C"),
               )
    .set_global_opts(
        # title_opts=opts.TitleOpts(title="f1", pos_top="48%"),
        legend_opts=opts.LegendOpts(pos_top="50%", pos_left="80%", textstyle_opts=opts.TextStyleOpts(font_size=20, font_family="Times New Roman")),
        xaxis_opts=opts.AxisOpts(type_="category", axislabel_opts=opts.LabelOpts(font_size=20, font_family="Times New Roman"), boundary_gap=False),
        yaxis_opts=opts.AxisOpts(name="f1", type_="value", position="left",# max_=15,
                                 axislabel_opts=opts.LabelOpts(font_size=20, font_family="Times New Roman"),
                                 name_textstyle_opts=opts.TextStyleOpts(font_size=20, font_family="Times New Roman")),
    )
)

grid = (
    Grid(init_opts=opts.InitOpts(width="900px", height="1200px"))
    .add(bar, grid_opts=opts.GridOpts(pos_top="15%", pos_bottom="60%", pos_right="10%"))
    .add(line, grid_opts=opts.GridOpts(pos_top="50%", pos_bottom="25%",  pos_left="10%"))
    .render("grid_vertical.html")
)
# make_snapshot(snapshot, grid.render, "measure_threshold.jpg")