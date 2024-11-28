import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from evaluation import evaluation


def save_evaluation_metrics_to_arrays(test_y, test_pred, binary_pred, metric):
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

    # 计算 ROC 曲线的 FPR 和 TPR
    fpr, tpr, thresholds = roc_curve(test_y, test_pred)
    roc_auc = round(auc(fpr, tpr), 5)

    # 打印 FPR 和 TPR 以便调试
    print(f"FPR: {fpr}, TPR: {tpr}, ROC AUC: {roc_auc}")

    mean_fpr = np.linspace(0, 1, 100)
    metric["auc_all"].append(roc_auc)
    metric["tpr_all"].append(np.interp(mean_fpr, fpr, tpr))
    metric["tpr_all"][-1][0] = 0.0  # 将第一个点设置为 0

    # 计算每个子数据集的 TPR@10%FPR 并保存到列表
    index_10_fpr = np.where(fpr >= 0.1)[0][0]  # 找到接近 10% FPR 的位置
    tpr_at_10_fpr = tpr[index_10_fpr]  # 对应的 TPR 值
    metric["tpr_at_10_fpr_all"].append(round(tpr_at_10_fpr, 5))


def main():
    model_name_list = ["deepwalk", "node2vec", "GCN", "GAT", "GraphSAGE", "ZipZap"]
    model_name = 'GNN + Multi-Agent'
    threshold_list = ["Round-off", "Custom", "Optimal", "TPR@10%FPR"]
    threshold = "TPR@10%FPR"
    if threshold == "Custom":
        threshold_Custom = 0.8
    else:
        threshold_Custom = ""

    time_range = {"MulDiGraph": ["2015-08-07", "2019-01-19"], "ZipZap": ["2017-06-24", "2022-03-01"]}
    dataset_name = "MulDiGraph"
    total_partitions = 15
    delay = "5"
    result_dir = f'../result/{dataset_name}/delay_{delay}/{model_name}/{threshold}{threshold_Custom}'

    metric = {"tpr_all": [], "precision_all": [], "recall_all": [], "f1_all": [], "tpr_at_10_fpr_all": [],
              "auc_all": [], "conf_matrix_all": []}

    for i in range(2, 16):
        result_model = pd.read_csv(f'../result/{dataset_name}/delay_{delay}/{model_name}/result_{str(i)}.csv')

        pred = result_model[['id', 'true_label', 'predicted_prob']]

        test_y = pred['true_label']
        test_pred = pred['predicted_prob']

        if threshold == "Round-off":
            # 将预测概率转换为二进制标签
            binary_pred = np.around(test_pred, 0).astype(int)
        elif threshold == "Custom":
            # 选择自定义阈值来生成二进制标签
            binary_pred = (test_pred >= threshold_Custom).astype(int)
        elif threshold == "Optimal":
            # 计算 FPR, TPR 和 thresholds
            fpr, tpr, thresholds = roc_curve(test_y, test_pred)
            # 计算 Youden's J statistic: J = Sensitivity + Specificity - 1
            J = tpr - fpr
            ix = np.argmax(J)  # 找到 J 最大值对应的索引
            best_threshold = thresholds[ix]
            print(f"Best threshold based on ROC curve: {best_threshold}")
            # 使用最佳阈值将预测概率转换为二进制标签
            binary_pred = (test_pred >= best_threshold).astype(int)
        elif threshold == "TPR@10%FPR":
            # 计算 FPR, TPR 和 thresholds
            fpr, tpr, thresholds = roc_curve(test_y, test_pred)
            # 找到 FPR >= 10% 的第一个索引
            index_10_fpr = np.where(fpr >= 0.1)[0][0]
            # 获取对应的阈值
            threshold_at_10_fpr = thresholds[index_10_fpr]
            print(f"Threshold at 10% FPR: {threshold_at_10_fpr}")
            # 使用该阈值将预测概率转换为二进制标签
            binary_pred = (test_pred >= threshold_at_10_fpr).astype(int)
        else:
            raise Exception("Not match threshold !!!")

        save_evaluation_metrics_to_arrays(test_y, test_pred, binary_pred, metric)

    start_time = time_range.get(dataset_name)[0]
    end_time = time_range.get(dataset_name)[1]
    evaluation(result_dir, metric, dataset_name, model_name, delay, total_partitions, start_time, end_time)


if __name__ == '__main__':
    main()
