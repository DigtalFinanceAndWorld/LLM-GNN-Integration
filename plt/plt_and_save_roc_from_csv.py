import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import ast


# 定义读取 CSV 文件并绘制 ROC 曲线的函数
def plot_and_save_roc(result_dir, model_name, total_partitions, start_time, end_time):
    # 读取 CSV 文件
    data = pd.read_csv(f"{result_dir}/result.csv").iloc[:total_partitions, :]

    # 解析并提取 TPR 和 AUC 信息
    tpr_all = []
    auc_all = []
    for index, row in data.iterrows():
        tpr_str = row['tpr_list']
        try:
            tpr_values = [float(num) for num in tpr_str.strip('[]').split()]
            tpr_all.append(tpr_values)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing TPR values for row {index}: {e}")
            continue

        # 获取 AUC 值
        auc_all.append(row['AUC'])

    # 将 TPR 列表转换为 numpy 数组，便于后续计算
    tpr_all = np.array(tpr_all)

    # 计算 Mean TPR 和 Mean FPR（这里假设 FPR 的采样点是均匀分布的）
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tpr_all, axis=0)

    # 检查 mean_tpr 是否为空或存在错误
    if mean_tpr.size == 0:
        print("Mean TPR is empty. Unable to plot ROC curve.")
        return

    # 将 mean_tpr 转换为一维数组，确保与 mean_fpr 的形状一致
    mean_tpr = mean_tpr.flatten()  # 或者使用 mean_tpr = mean_tpr.squeeze()

    mean_tpr[-1] = 1.0  # 最后一个点设置为 1，保证 ROC 曲线的完整性

    # 保存 Mean ROC 数据到 .npy 文件
    mean_roc_data_file = f'{result_dir}/mean_roc.npy'
    np.save(mean_roc_data_file, mean_tpr)

    # 计算 AUC 的统计数据
    auc_all = np.array(auc_all)
    auc_stats = {
        "Mean": np.mean(auc_all),
        "Std": np.std(auc_all),
        "Max": np.max(auc_all),
        "Min": np.min(auc_all),
        "Median": np.median(auc_all)
    }

    # 保存 AUC 和其他统计数据到 .csv 文件
    mean_roc_csv_file = f'{result_dir}/mean_roc.csv'
    with open(mean_roc_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Mean AUC", "AUC Std", "Max AUC", "Min AUC", "Median AUC"])
        writer.writerow([auc_stats["Mean"], auc_stats["Std"], auc_stats["Max"], auc_stats["Min"], auc_stats["Median"]])

    # 绘制 ROC 曲线
    plt.figure(figsize=(10, 8))
    for i, tpr in enumerate(tpr_all):
        plt.plot(mean_fpr, tpr.flatten(), lw=1, alpha=0.3)

    plt.plot([], [], ' ', label=f'Total Partitions: {total_partitions}')
    plt.plot([], [], ' ', label=f'Start Date: {start_time}')
    plt.plot([], [], ' ', label=f'End Date: {end_time}')
    plt.plot([], [], ' ', label=f'Min AUC: {auc_stats["Min"]:.2f}')
    plt.plot([], [], ' ', label=f'Max AUC: {auc_stats["Max"]:.2f}')
    plt.plot([], [], ' ', label=f'Median AUC: {auc_stats["Median"]:.2f}')
    plt.plot([], [], ' ', label=f'Mean AUC: {auc_stats["Mean"]:.2f}')
    plt.plot([], [], ' ', label=f'Std AUC: {auc_stats["Std"]:.2f}')
    plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=.8,
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc_stats["Mean"], auc_stats["Std"]))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right", fontsize=10, frameon=True)

    # 保存 ROC 曲线到文件
    plt_png_file = f'{result_dir}/roc_curve.png'
    plt.savefig(plt_png_file, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'ROC Curve and statistics saved for {model_name}.')


def main():
    model_name = 'MulDiGraph'

    baseline_list = ["deepwalk_15", "GAT_15", "GCN_15", "GraphSAGE_15", "node2vec_15", "ZipZap_15"]
    for baseline in baseline_list:
        result_dir = f'/home/a/zmb_workspace/product/Phisher_detect/baselines/result/{model_name}/delay_5/{baseline}'
        total_partitions = 15
        start_time = "2015-08-07"
        end_time = "2019-01-19"
        plot_and_save_roc(result_dir, model_name, total_partitions, start_time, end_time)


if __name__ == '__main__':
    main()
