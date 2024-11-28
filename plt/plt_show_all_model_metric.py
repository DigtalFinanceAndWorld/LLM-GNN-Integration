import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # dataset_name = "MulDiGraph"
    dataset_name = "MulDiGraph"
    total_partitions = 18
    result_dir = f"/home/a/zmb_workspace/product/Phisher_detect/baselines/result/{dataset_name}/delay_5"
    # model_name_list = ["deepwalk", "node2vec", "GCN", "GAT", "GraphSAGE", "ZipZap",
    #                    "ensemble_GCN_llama_no_finetune_recent_1000", "ensemble_GCN_llama_no_finetune_represent_1000"]
    model_name_list = ["deepwalk", "node2vec", "GCN", "GAT", "GraphSAGE", "ZipZap", "ensemble_GCN_gpt-4o-mini_500_1000_2000"]
    metrics = ["Precision", "Recall", "F1", "TPR@10%FPR", "AUC"]
    all_data = {metric: {} for metric in metrics}

    for model_name in model_name_list:
        result_csv_file = f"{result_dir}/{model_name}/result.csv"
        if os.path.exists(result_csv_file):
            data = pd.read_csv(result_csv_file).iloc[:total_partitions, :]

            for index, row in data.iterrows():
                for metric in metrics:
                    # 如果当前行存在该指标，则将其值存储到 all_data 中
                    if metric in row:
                        # 将 metric 的值添加到相应模型的指标列表中
                        if model_name not in all_data[metric]:
                            all_data[metric][model_name] = []
                        metric_str = row[metric]
                        all_data[metric][model_name].append(float(metric_str))

    for metric, data in all_data.items():
        plt.figure(figsize=(20, 6))
        for model, values in data.items():
            plt.plot(range(1, len(values) + 1), values, label=model, marker='o')
        plt.title(f'Comparison of {metric} Across Models')
        plt.xlabel('Data_num')
        plt.ylabel(metric)
        plt.legend()

        save_path = os.path.join(result_dir, f"{metric}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {metric} comparison plot at: {save_path}")

    print("All comparison plots have been saved successfully.")


if __name__ == '__main__':
    main()
