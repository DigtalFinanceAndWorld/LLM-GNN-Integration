import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import csv
import os

from plt_show_metric import plt_draw

mean_fpr = np.linspace(0, 1, 100)


def save_evaluation_metrics_to_arrays(test_y, test_pred, metric):
    # 将预测概率转换为二进制标签
    binary_pred = np.around(test_pred, 0).astype(int)

    # 选择自定义阈值来生成二进制标签
    # threshold = 0.8
    # binary_pred = (test_pred >= threshold).astype(int)

    # # 计算 FPR, TPR 和 thresholds
    # fpr, tpr, thresholds = roc_curve(test_y, test_pred)
    #
    # # 计算 Youden's J statistic: J = Sensitivity + Specificity - 1
    # J = tpr - fpr
    # ix = np.argmax(J)  # 找到 J 最大值对应的索引
    # best_threshold = thresholds[ix]
    # print(f"Best threshold based on ROC curve: {best_threshold}")
    #
    # # 使用最佳阈值将预测概率转换为二进制标签
    # binary_pred = (test_pred >= best_threshold).astype(int)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(test_y, binary_pred)

    # 判断特殊情况：所有标签和预测均为 0 时，混淆矩阵需特殊处理
    if np.all(test_y == 0) and np.all(binary_pred == 0):
        TP = conf_matrix[0][0]
        FN = FP = TN = 0
    else:
        TP, FN, FP, TN = conf_matrix.ravel()

    # 打印并保存混淆矩阵结果
    matrix_output = f"TP={TP}, FN={FN}, FP={FP}, TN={TN}"
    print(matrix_output)
    metric["conf_matrix_all"].append(matrix_output)

    # 计算 Precision, Recall 和 F1 Score
    precision = precision_score(test_y, binary_pred)
    recall = recall_score(test_y, binary_pred)
    f1 = f1_score(test_y, binary_pred)

    # 保存评估指标到相应数组
    metric["precision_all"].append(round(precision, 5))
    metric["recall_all"].append(round(recall, 5))
    metric["f1_all"].append(round(f1, 5))

    # # 判断 test_y 是否只有一种标签（全为 0 或全为 1），或 test_pred 是否只有一个唯一值
    # if FP + TN == 0:
    #     print("Warning: No true negative samples. Setting ROC AUC to 0.5 (random guess).")
    #     fpr, tpr, roc_auc = np.array([0, 1]), np.array([0, 1]), 0.5
    # else:
    #
    # 计算 ROC 曲线的 FPR 和 TPR
    fpr, tpr, thresholds = roc_curve(test_y, test_pred)
    roc_auc = round(auc(fpr, tpr), 5)

    # 打印 FPR 和 TPR 以便调试
    print(f"FPR: {fpr}, TPR: {tpr}, ROC AUC: {roc_auc}")

    metric["auc_all"].append(roc_auc)
    metric["tpr_all"].append(np.interp(mean_fpr, fpr, tpr))
    metric["tpr_all"][-1][0] = 0.0  # 将第一个点设置为 0

    # 计算每个子数据集的 TPR@10%FPR 并保存到列表
    index_10_fpr = np.where(fpr >= 0.1)[0][0]  # 找到接近 10% FPR 的位置
    tpr_at_10_fpr = tpr[index_10_fpr]  # 对应的 TPR 值
    metric["tpr_at_10_fpr_all"].append(round(tpr_at_10_fpr, 5))


def calculate_statistics(data_list):
    if len(data_list) == 0:
        return {}

    max_val = round(max(data_list), 5)
    min_val = round(min(data_list), 5)
    median_val = round(np.median(data_list), 5)
    mean_val = round(np.mean(data_list, axis=0), 5)
    std_val = round(np.std(data_list), 5)
    sharpe_ratio = round(mean_val / std_val, 5) if std_val != 0 else np.nan

    # 将结果存储到字典中
    statistics_dict = {
        "Max": max_val,
        "Min": min_val,
        "Median": median_val,
        "Mean": mean_val,
        "Std": std_val,
        "Sharpe Ratio": sharpe_ratio
    }

    return statistics_dict


def evaluation(result_dir, metric, dataset_name, model_name, delay, total_partitions, start_time, end_time):
    os.makedirs(result_dir, exist_ok=True)

    precision_stats = calculate_statistics(metric["precision_all"])
    recall_stats = calculate_statistics(metric["recall_all"])
    f1_stats = calculate_statistics(metric["f1_all"])
    tpr_at_10_fpr_stats = calculate_statistics(metric["tpr_at_10_fpr_all"])
    auc_stats = calculate_statistics(metric["auc_all"])

    # 结果的保存文件路径
    result_csv_file = f'{result_dir}/result.csv'
    plt_png_file = f'{result_dir}/roc_curve.png'

    # 准备数据列表，包括标题行
    data = [["Data_num", "Precision", "Recall", "F1", "TPR@10%FPR", "AUC", "conf_matrix", "tpr_list"]]

    # 添加每一行的数据
    for i in range(len(metric["precision_all"])):
        data.append([i + 1, metric["precision_all"][i], metric["recall_all"][i], metric["f1_all"][i],
                     metric["tpr_at_10_fpr_all"][i], metric["auc_all"][i], metric["conf_matrix_all"][i], metric["tpr_all"][i]])

    # 将数据列表写入 CSV 文件
    with open(result_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    data = [
        ["Max", precision_stats["Max"], recall_stats["Max"], f1_stats["Max"], tpr_at_10_fpr_stats["Max"],
         auc_stats["Max"]],
        ["Min", precision_stats["Min"], recall_stats["Min"], f1_stats["Min"], tpr_at_10_fpr_stats["Min"],
         auc_stats["Min"]],
        ["Median", precision_stats["Median"], recall_stats["Median"], f1_stats["Median"],
         tpr_at_10_fpr_stats["Median"], auc_stats["Median"]],
        ["Mean", precision_stats["Mean"], recall_stats["Mean"], f1_stats["Mean"], tpr_at_10_fpr_stats["Mean"],
         auc_stats["Mean"]],
        ["Std", precision_stats["Std"], recall_stats["Std"], f1_stats["Std"], tpr_at_10_fpr_stats["Std"],
         auc_stats["Std"]],
        ["Sharpe Ratio", precision_stats["Sharpe Ratio"], recall_stats["Sharpe Ratio"],
         f1_stats["Sharpe Ratio"],
         tpr_at_10_fpr_stats["Sharpe Ratio"], auc_stats["Sharpe Ratio"]]]

    # 将数据列表写入 CSV 文件
    with open(result_csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    # 保存 Mean ROC 数据到 .npy 文件
    mean_tpr = np.mean(metric["tpr_all"], axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_data_file = f'{result_dir}/mean_roc.npy'
    np.save(mean_roc_data_file, mean_tpr)  # 保存 mean_tpr 数据

    # 保存 AUC 和其他统计数据到 .csv 文件
    mean_roc_csv_file = f'{result_dir}/mean_roc.csv'
    with open(mean_roc_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Mean AUC", "AUC Std", "Max AUC", "Min AUC", "Median AUC"])
        writer.writerow(
            [auc_stats["Mean"], auc_stats["Std"], auc_stats["Max"], auc_stats["Min"], auc_stats["Median"]])

    # 绘制 ROC 曲线
    plt.figure(figsize=(10, 8))
    for i, tpr in enumerate(metric["tpr_all"]):
        plt.plot(mean_fpr, tpr, lw=1, alpha=0.3)

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
    plt.legend(loc="lower right", bbox_to_anchor=(1, 0), fontsize=10, frameon=True)
    plt.savefig(plt_png_file, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"The result CSV file: {result_csv_file}, plt png file: {plt_png_file}")
    plt_draw(result_dir)
    print("Finished !!!")
