import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from evaluation import *


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def check_valid(x):
    try:
        return float(x)
    except ValueError:
        return 'Invalid'


def main():
    # strategy_list = ["head", "recent", "represent", "represent_gpt]
    time_range = {"MulDiGraph": ["2015-08-07", "2019-01-19"], "ZipZap": ["2017-06-24", "2022-03-01"]}
    dataset_name = "ZipZap"
    total_partitions = 11
    delay = "5"
    multiple = 100
    strategy = "represent"
    finetune = "llama_no_finetune"
    # finetune = "llama_finetune"
    gnn_model = "GCN"

    model_name = f'ensemble_{gnn_model}_{finetune}_{strategy}_{multiple}'
    result_dir = f'../result/{dataset_name}/delay_{delay}/{model_name}'

    metric = {"tpr_all": [], "precision_all": [], "recall_all": [], "f1_all": [], "tpr_at_10_fpr_all": [],
              "auc_all": [], "conf_matrix_all": []}

    for i in range(1, total_partitions + 1):
        result_llm = pd.read_excel(
            f'../../prompt/result/{finetune}/{dataset_name}/{multiple}/delay_{delay}/{strategy}/{str(i)}.xlsx')
        result_gnn = pd.read_csv(f'../result/{dataset_name}/delay_{delay}/{gnn_model}/result_{str(i)}.csv')
        node_map = json.load(
            open(f'../../dataset/clustered_graphs/{dataset_name}/{multiple}/delay_{delay}/{str(i)}/test_nodemap.json'))
        output_csv_path = f'../result/{dataset_name}/delay_{delay}/{model_name }/result_{str(i)}.csv'

        result_gnn.columns = ['id', 'true_label', 'gnn_pred']
        pred_gnn = result_gnn[['id', 'gnn_pred', 'true_label']]

        pred_gnn['gnn_pred'] = normalize(pred_gnn['gnn_pred'])
        pred_llm = result_llm[['node_address', 'score']]
        pred_llm.columns = ['cid', 'llm_pred']
        pred_llm['llm_pred'] = pred_llm['llm_pred'].apply(lambda x: check_valid(x))
        pred_llm = pred_llm[(pred_llm['llm_pred'] != 'Invalid')]
        pred_llm = pred_llm[(pred_llm['llm_pred'] >= 0) & (pred_llm['llm_pred'] <= 100)].reset_index(drop=True)
        pred_llm['cid'] = pred_llm['cid'].apply(lambda x: int(x.split('_')[1]))
        pred_llm['llm_pred'] = normalize(pred_llm['llm_pred'])

        pred_gnn['cid'] = pred_gnn['id'].apply(lambda x: node_map[x])
        pred = pd.merge(pred_gnn, pred_llm, on='cid', how='inner')
        pred = pred[['id', 'gnn_pred', 'llm_pred', 'true_label']]
        pred['predicted_prob'] = normalize(pred['gnn_pred'] + pred['llm_pred'])

        if not os.path.exists(os.path.dirname(output_csv_path)):
            os.makedirs(os.path.dirname(output_csv_path))
        pred.to_csv(output_csv_path, index=False)

        save_evaluation_metrics_to_arrays(pred['true_label'], pred['predicted_prob'], metric)

    start_time = time_range.get(dataset_name)[0]
    end_time = time_range.get(dataset_name)[1]
    evaluation(result_dir, metric, dataset_name, model_name, delay, total_partitions, start_time, end_time)


if __name__ == '__main__':
    main()
