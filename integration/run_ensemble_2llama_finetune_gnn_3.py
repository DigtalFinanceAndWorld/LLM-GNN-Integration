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
    # strategy_list = ["head", "recent", "represent", "history_recent", "purest"]
    time_range = {"MulDiGraph": ["2015-08-07", "2019-01-19"], "ZipZap": ["2017-06-24", "2022-03-01"]}
    dataset_name = "MulDiGraph"
    total_partitions = 15
    delay = "5"
    multiple1 = "100"
    multiple2 = "500"
    strategy = "purest"
    agent = "single-expert"
    finetune = "llama_finetune"
    finetune_type = "finetune_by_label_analysis_logits_smote"
    gnn_model = "GraphSAGE"

    model_name = f'ensemble_{gnn_model}_{finetune}_{finetune_type}_{strategy}_{multiple1}_{multiple2}_{agent}'
    result_dir = f'../result/{dataset_name}/delay_{delay}/{model_name}'

    metric = {"tpr_all": [], "precision_all": [], "recall_all": [], "f1_all": [], "tpr_at_10_fpr_all": [],
              "auc_all": [], "conf_matrix_all": []}

    for i in range(2, total_partitions + 1):
        result_llm_1 = pd.read_excel(
            f'../../prompt/result/{finetune}/{finetune_type}/{agent}/{dataset_name}/{multiple1}/delay_{delay}/{strategy}/{str(i)}.xlsx')
        result_llm_2 = pd.read_excel(
            f'../../prompt/result/{finetune}/{finetune_type}/{agent}/{dataset_name}/{multiple2}/delay_{delay}/{strategy}/{str(i)}.xlsx')

        result_gnn = pd.read_csv(f'../result/{dataset_name}/delay_{delay}/{gnn_model}/result_{str(i)}.csv')
        node_map_1 = json.load(
            open(f'../../dataset/clustered_graphs/{dataset_name}/{multiple1}/delay_{delay}/{str(i)}/test_nodemap.json'))
        node_map_2 = json.load(
            open(f'../../dataset/clustered_graphs/{dataset_name}/{multiple2}/delay_{delay}/{str(i)}/test_nodemap.json'))
        output_csv_path = f'{result_dir}/result_{str(i)}.csv'

        result_gnn.columns = ['id', 'true_label', 'gnn_pred']
        pred_gnn = result_gnn[['id', 'gnn_pred', 'true_label']]
        pred_gnn['gnn_pred'] = normalize(pred_gnn['gnn_pred'])

        pred_llm_1 = result_llm_1[['node_address', 'label', 'confidence_score']]
        pred_llm_1['llm_pred'] = pred_llm_1.apply(
            lambda x: x['label'] * x['confidence_score'] + (1 - x['label']) * (1 - x['confidence_score']), axis=1
        )
        pred_llm_1 = pred_llm_1[['node_address', 'llm_pred']]
        pred_llm_1.columns = ['cid1', 'llm_pred_1']
        pred_llm_1['llm_pred_1'] = pred_llm_1['llm_pred_1'].apply(lambda x: check_valid(x))
        pred_llm_1 = pred_llm_1[(pred_llm_1['llm_pred_1'] != 'Invalid')]
        pred_llm_1 = pred_llm_1[(pred_llm_1['llm_pred_1'] >= 0) & (pred_llm_1['llm_pred_1'] <= 100)].reset_index(
            drop=True)
        pred_llm_1['cid1'] = pred_llm_1['cid1'].apply(lambda x: int(x.split('_')[1]))
        pred_llm_1['llm_pred_1'] = normalize(pred_llm_1['llm_pred_1'])

        pred_llm_2 = result_llm_2[['node_address', 'label', 'confidence_score']]
        pred_llm_2['llm_pred'] = pred_llm_2.apply(
            lambda x: x['label'] * x['confidence_score'] + (1 - x['label']) * (1 - x['confidence_score']), axis=1
        )
        pred_llm_2 = pred_llm_2[['node_address', 'llm_pred']]
        pred_llm_2.columns = ['cid2', 'llm_pred_2']
        pred_llm_2['llm_pred_2'] = pred_llm_2['llm_pred_2'].apply(lambda x: check_valid(x))
        pred_llm_2 = pred_llm_2[(pred_llm_2['llm_pred_2'] != 'Invalid')]
        pred_llm_2 = pred_llm_2[(pred_llm_2['llm_pred_2'] >= 0) & (pred_llm_2['llm_pred_2'] <= 100)].reset_index(
            drop=True)
        pred_llm_2['cid2'] = pred_llm_2['cid2'].apply(lambda x: int(x.split('_')[1]))
        pred_llm_2['llm_pred_2'] = normalize(pred_llm_2['llm_pred_2'])


        pred_gnn['cid1'] = pred_gnn['id'].apply(lambda x: node_map_1[x])
        pred_gnn['cid2'] = pred_gnn['id'].apply(lambda x: node_map_2[x])

        pred = pd.merge(pred_gnn, pred_llm_1, on='cid1', how='inner')
        pred = pd.merge(pred, pred_llm_2, on='cid2', how='inner')

        pred = pred[['id', 'gnn_pred', 'llm_pred_1', 'llm_pred_2', 'true_label', 'cid1', 'cid2']]

        weight1 = normalize(
            pred.groupby('cid1').apply(lambda x: x['gnn_pred'].sort_values(ascending=False)[:20].mean())).to_dict()
        weight2 = normalize(
            pred.groupby('cid2').apply(lambda x: x['gnn_pred'].sort_values(ascending=False)[:20].mean())).to_dict()
        print(weight1)
        print(weight2)
        pred['llm_weight1'] = pred['cid1'].map(weight1)
        pred['llm_weight2'] = pred['cid2'].map(weight2)
        pred['predicted_prob'] = (pred['gnn_pred'] + pred['llm_pred_1'] * pred['llm_weight1'] + pred['llm_pred_2'] *
                                  pred['llm_weight2']) / (1 + pred['llm_weight1'] + pred['llm_weight2'])

        if not os.path.exists(os.path.dirname(output_csv_path)):
            os.makedirs(os.path.dirname(output_csv_path))
        pred.to_csv(output_csv_path, index=False)

        save_evaluation_metrics_to_arrays(pred['true_label'], pred['predicted_prob'], metric)

    start_time = time_range.get(dataset_name)[0]
    end_time = time_range.get(dataset_name)[1]
    evaluation(result_dir, metric, dataset_name, model_name, delay, total_partitions, start_time, end_time)


if __name__ == '__main__':
    main()
