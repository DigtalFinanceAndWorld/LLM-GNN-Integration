import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import csv


def cal_evaluation_metrics(test_y, test_pred):
    binary_pred = np.around(test_pred, 0).astype(int)

    conf_matrix = confusion_matrix(test_y, binary_pred)
    print(f"test_y: {test_y}, binary_pred: {binary_pred}, conf_matrix: {conf_matrix}")

    if np.all(test_y == 0) and np.all(binary_pred == 0):
        TP = conf_matrix[0][0]
        FN = FP = TN = 0
    else:
        TP, FN, FP, TN = conf_matrix.ravel()

    matrix_output = f"TP={TP}, FN={FN}, FP={FP}, TN={TN}"
    precision = precision_score(test_y, binary_pred)
    recall = recall_score(test_y, binary_pred)
    f1 = f1_score(test_y, binary_pred)
    fpr, tpr, thresholds = roc_curve(test_y, test_pred)
    tpr_list = []
    mean_fpr = np.linspace(0, 1, 100)
    tpr_list.append(np.interp(mean_fpr, fpr, tpr))
    tpr_list[-1][0] = 0.0  # 将第一个点设置为 0
    index_10_fpr = np.where(fpr >= 0.1)[0][0]
    tpr_at_10_fpr = tpr[index_10_fpr]

    return (round(precision, 5), round(recall, 5), round(f1, 5), round(tpr_at_10_fpr, 5),
            round(auc(fpr, tpr), 5), matrix_output, tpr_list)


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


def create_csv(test_y, test_pred, result_csv):
    precision, recall, f1, tpr_at_10_fpr, auc, matrix_output, tpr_list = cal_evaluation_metrics(test_y, test_pred)
    data = [["Data_num", "Precision", "Recall", "F1", "TPR@10%FPR", "AUC", "conf_matrix", "TPR"],
            ["1", precision, recall, f1, tpr_at_10_fpr, auc, matrix_output, tpr_list]]
    with open(result_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"The result CSV file: {result_csv}")


def add_data(test_y, test_pred, result_csv, index):
    precision, recall, f1, tpr_at_10_fpr, auc, matrix_output, tpr_list = cal_evaluation_metrics(test_y, test_pred)
    data = [[index, precision, recall, f1, tpr_at_10_fpr, auc, matrix_output, tpr_list]]
    with open(result_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"The result CSV file: {result_csv}")


def add_and_evaluation(test_y, test_pred, result_csv, index):
    precision, recall, f1, tpr_at_10_fpr, auc, matrix_output, tpr_list = cal_evaluation_metrics(test_y, test_pred)
    data = [[index, precision, recall, f1, tpr_at_10_fpr, auc, matrix_output, tpr_list]]

    with open(result_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    metric = {}
    with open(result_csv, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        data_dict = {key: [] for key in header[1:]}
        for row in reader:
            for i, value in enumerate(row[1:]):
                data_dict[header[i + 1]].append(value)

    precision_stats = calculate_statistics(metric["precision"])
    recall_stats = calculate_statistics(metric["recall"])
    f1_stats = calculate_statistics(metric["f1"])
    tpr_at_10_fpr_stats = calculate_statistics(metric["tpr_at_10_fpr"])
    auc_stats = calculate_statistics(metric["auc"])

    evl_data = [
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

    with open(result_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(evl_data)

    print(f"The result CSV file: {result_csv}")
    print("Finished !!!")
