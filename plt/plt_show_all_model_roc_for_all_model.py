import csv
import os

import numpy as np
from matplotlib import pyplot as plt


def load_mean_roc_data(mean_roc_file, auc_csv_file):
    data = {}
    try:
        data['mean_tpr'] = np.load(mean_roc_file)
    except FileNotFoundError:
        print(f"Error: The file {mean_roc_file} was not found.")
        data['mean_tpr'] = None

    try:
        with open(auc_csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行
            auc_row = next(reader)  # 读取第一行内容
            auc_stats = {
                "Mean": float(auc_row[0]),
                "Std": float(auc_row[1]),
                "Max": float(auc_row[2]),
                "Min": float(auc_row[3]),
                "Median": float(auc_row[4])
            }
        data['auc_stats'] = auc_stats
    except FileNotFoundError:
        print(f"Error: The file {auc_csv_file} was not found.")
        data['auc_stats'] = {}

    return data


def plot_all_models_mean_roc(mean_roc_data, result_dir):
    plt.figure(figsize=(12, 10))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 颜色列表，可以根据模型数量选择更多颜色

    for idx, (model_name, mean_tpr) in enumerate(mean_roc_data.items()):
        if "_stats" in model_name:  # 跳过统计数据
            continue

        # 获取 AUC 统计信息
        auc_stats = mean_roc_data.get(model_name + "_stats", {})
        mean_auc = auc_stats.get("Mean", 0)
        std_auc = auc_stats.get("Std", 0)

        # 绘制每个模型的 Mean ROC 曲线
        plt.plot(np.linspace(0, 1, 100), mean_tpr, color=colors[idx % len(colors)],
                 lw=2, alpha=.8, label=f'{model_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curves for All Methods')
    plt.legend(loc="lower right")

    # 保存最终的图像
    combined_roc_file = os.path.join(result_dir, 'Mean ROC Curves for All Methods.png')
    # combined_roc_file = os.path.join(result_dir, 'Mean ROC Curves of Different Multiple Fusion.png')
    plt.savefig(combined_roc_file, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined Mean ROC curve saved to: {combined_roc_file}")


def main():
    # 初始化字典来存储所有模型的 Mean ROC 数据
    all_models_mean_roc_data = {}

    # dataset_name = "MulDiGraph"
    dataset_name = "MulDiGraph"
    result_dir = f"/home/a/zmb_workspace/product/Phisher_detect/baselines/result/{dataset_name}/delay_5"

    model_name_list = ["deepwalk", "GAT", "GCN", "GraphSAGE", "node2vec", "ZipZap",
                       "GNN + Single-Expert", "GNN + Multi-Agent", "GNN + BC-FT", "GNN + SRA-FT", "GNN + CRRA-FT"]

    # model_name_list = ["ensemble_500x", "ensemble_1000x", "ensemble_2000x", "ensemble_500x_1000x", "ensemble_500x_2000x",
    #                    "ensemble_1000x_2000x", "ensemble_500x_1000x_2000x"]
    for model_name in model_name_list:
        mean_roc_file = f"{result_dir}/{model_name}/mean_roc.npy"
        auc_csv_file = f"{result_dir}/{model_name}/mean_roc.csv"

        model_data = load_mean_roc_data(mean_roc_file, auc_csv_file)
        all_models_mean_roc_data[model_name] = model_data['mean_tpr']
        all_models_mean_roc_data[model_name + "_stats"] = model_data['auc_stats']

    # 打印所有模型的数据以确认加载成功
    print("All models' Mean ROC and AUC statistics loaded successfully:")
    for model, mean_tpr in all_models_mean_roc_data.items():
        if "_stats" not in model:
            print(f"{model}: Mean AUC = {all_models_mean_roc_data[model + '_stats'].get('Mean', 'N/A')}")

    # 如果需要，可以将 all_models_mean_roc_data 传递给一个函数来绘制所有模型的 ROC 曲线
    plot_all_models_mean_roc(all_models_mean_roc_data, result_dir)


if __name__ == "__main__":
    main()
