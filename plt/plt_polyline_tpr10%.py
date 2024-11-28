import matplotlib.pyplot as plt

# 数据：三个模型在7个分区上的AUC值
partitions = [4, 5, 6, 7, 8, 9, 10]
llama_1 = [0.62338, 0.57317, 0.57895, 0.47205, 0.36232, 0.78512, 0.72531]  # LLaMA 微调的融合 AUC
gpt_1 = [0.32468, 0.46341, 0.54386, 0.32298, 0.18841, 0.64463, 0.62428]  # 一个倍数 GPT 的融合 AUC
gpt_3 = [0.5974, 0.65854, 0.51754, 0.3913, 0.3913, 0.69008, 0.78613]  # 三个倍数 GPT 的融合 AUC

# 创建折线图
plt.figure(figsize=(10, 6))

# 绘制三条折线，分别代表不同模型的AUC变化
plt.plot(partitions, llama_1, marker='o', linestyle='-', color='b', linewidth=2, markersize=8, label='LLaMA Finetune (500)')
plt.plot(partitions, gpt_1, marker='s', linestyle='--', color='g', linewidth=2, markersize=8, label='GPT Ensemble (500)')
plt.plot(partitions, gpt_3, marker='^', linestyle='-.', color='r', linewidth=2, markersize=8, label='GPT Ensemble (500, 1000, 2000)')

# 添加标题和坐标轴标签
plt.title('TPR@10%FPR Trend for GPT and LLaMA Models', fontsize=18)
plt.xlabel('Partition', fontsize=14)
plt.ylabel('TPR@10%FPR', fontsize=14)

# 设置x轴范围和刻度
plt.xticks(partitions)
plt.ylim(0.1, 0.8)

# 显示图例，并将其放置在右下角
plt.legend(loc='lower left', fontsize=12)

# 显示网格
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图片
plt.savefig('TPR@10%FPR_trend_gpt_llama.png', format='png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
