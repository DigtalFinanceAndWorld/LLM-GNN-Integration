import matplotlib.pyplot as plt

# 数据：三个模型在15个分区上的AUC值
partitions = ["2017-08", "2017-09", "2017-10", "2017-11", "2017-12",
              "2018-01", "2018-02", "2018-03", "2018-04", "2018-05",
              "2018-06", "2018-07", "2018-08", "2018-09", "2018-10"]

# 修正后的数据
zipzap = [
    0.69838, 0.86007, 0.767, 0.89642, 0.69555, 0.67067, 0.78411,
    0.71397, 0.71659, 0.88853, 0.56469, 0.75315, 0.60136, 0.66087, 0.78492
]

multi_agent = [
    0.92441, 0.92224, 0.88971, 0.88101, 0.89127, 0.89406, 0.85363,
    0.87019, 0.92492, 0.9214, 0.91097, 0.91826, 0.91502, 0.77008, 0.75398
]

crra_ft = [
    0.93011, 0.94091, 0.92973, 0.91658, 0.91848, 0.92819, 0.91542,
    0.91801, 0.93013, 0.94369, 0.93661, 0.91744, 0.90861, 0.83018, 0.85946
]

# 创建折线图
plt.figure(figsize=(10, 8.2))

# 绘制三条折线
plt.plot(partitions, zipzap, marker='o', linestyle='-', color='b', linewidth=2, markersize=8, label='ZipZap')
plt.plot(partitions, multi_agent, marker='s', linestyle='--', color='g', linewidth=2, markersize=8, label='GNN + Multi-Agent')
plt.plot(partitions, crra_ft, marker='^', linestyle='-.', color='r', linewidth=2, markersize=8, label='GNN + CRRA-FT')


# 添加标题和坐标轴标签
# plt.title('AUC Trend for GPT and LLaMA Models', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Mean(AUC)', fontsize=14)

# 设置x轴范围和刻度，旋转45度
plt.xticks(partitions, rotation=45, fontsize=10)
plt.ylim(0.5, 1.0)

# 显示图例，并将其放置在左下角
plt.legend(loc='lower left', fontsize=12)

# 显示网格
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图片
plt.savefig('AUC_trend.png', format='png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
