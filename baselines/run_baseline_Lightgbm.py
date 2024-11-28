import re

import matplotlib.pyplot as plt
from lightgbm.callback import early_stopping
import lightgbm as lgb
import multiprocessing
import random

from evaluation import evaluation, save_evaluation_metrics_to_arrays
from data_process import *

from base_deepwalk import run_deepwalk
from base_node2vec import run_node2vec
from base_gat import run_GAT
from base_gcn import run_GCN
from base_graphsage import run_GraphSAGE

random.seed(42)


def main():
    num_cores = multiprocessing.cpu_count()
    workers = num_cores - 5
    # dataset_name_list = ["MulDiGraph", "ZipZap"]
    dataset_name_list = ["MulDiGraph"]
    time_range = {"MulDiGraph": ["2015-08-07", "2019-01-19"], "ZipZap": ["2017-06-24", "2022-03-01"]}
    for dataset_name in dataset_name_list:
        delay_list = [5]
        # delay_list = [5, 15, 30]
        for delay in delay_list:
            dataset_dir = f'../../dataset/{dataset_name}/data/GNN/delay_{delay}'
            dataset_list = []
            if os.path.exists(dataset_dir):
                print(f"Dir exists: {dataset_dir}")
                dataset_list = os.listdir(dataset_dir)
            else:
                print(f"Dir does not exists: {dataset_dir}")

            def natural_sort_key(filename):
                return [int(s) if s.isdigit() else s for s in re.split('([0-9]+)', filename)]

            sorted_dataset_list = sorted(dataset_list, key=natural_sort_key)
            print(f"Dataset_list: {sorted_dataset_list}")

            model_functions = {
                # "GAT": run_GAT,
                "GCN": run_GCN,
                # "GraphSAGE": run_GraphSAGE,
                # "deepwalk": run_deepwalk,
                # "node2vec": run_node2vec,
                # "trans2vec": trans2vec_run,
                # "GCN_test": run_GCN_test
            }

            for model_name, model_func in model_functions.items():

                metric = {"tpr_all": [], "precision_all": [], "recall_all": [], "f1_all": [], "tpr_at_10_fpr_all": [],
                          "auc_all": [], "conf_matrix_all": []}


                result_dir = f'../result/{dataset_name}/delay_{delay}/{model_name}_2'
                os.makedirs(result_dir, exist_ok=True)

                for index in sorted_dataset_list:
                    embeddings_csv_dir = f'../data/{dataset_name}/{model_name}/delay_{delay}/{index}'
                    os.makedirs(embeddings_csv_dir, exist_ok=True)
                    print(
                        f"================== Training {model_name} for dataset delay_{delay} index: {index} ==================")

                    if model_name == "deepwalk":
                        dimensions = 64
                        walk_length = 40
                        num_walks = 4
                        window_size = 4
                        model_func(dataset_dir, embeddings_csv_dir, index, dimensions,
                                   walk_length, num_walks, window_size, workers)
                    elif model_name == "node2vec":
                        dimensions = 64
                        walk_length = 20
                        num_walks = 10
                        model_func(dataset_dir, embeddings_csv_dir, index, dimensions,
                                   walk_length, num_walks, workers)
                    elif model_name == "GCN":
                        input_feat_dim = 8
                        hidden_dim = 16
                        output_dim = 8
                        learning_rate = 0.01
                        num_epochs = 100
                        model_func(dataset_dir, embeddings_csv_dir, index, input_feat_dim, hidden_dim, output_dim, learning_rate, num_epochs)
                    elif model_name == "GAT":
                        input_feat_dim = 8
                        hidden_dim = 16
                        output_dim = 8
                        learning_rate = 0.01
                        num_epochs = 50
                        model_func(dataset_dir, embeddings_csv_dir, index, input_feat_dim, hidden_dim, output_dim, learning_rate, num_epochs)
                    elif model_name == "GraphSAGE":
                        in_channels = 8
                        hidden_channels = 64
                        out_channels = 8
                        learning_rate = 0.01
                        num_epochs = 100
                        model_func(dataset_dir, embeddings_csv_dir, index, in_channels, hidden_channels, out_channels, learning_rate, num_epochs)
                    elif model_name == "trans2vec":
                        model_func()
                    elif model_name == "GCN_test":
                        input_feat_dim = 8
                        hidden_dim = 16
                        output_dim = 8
                        learning_rate = 0.01
                        num_epochs = 100
                        model_func(dataset_dir, embeddings_csv_dir, index, input_feat_dim, hidden_dim, output_dim, learning_rate, num_epochs)

                    # LightGBM
                    train_x, train_y, val_x, val_y, test_x, test_y, test_id = read_csv_data(dataset_dir, embeddings_csv_dir, index)
                    train_matrix = lgb.Dataset(train_x, label=train_y)
                    valid_matrix = lgb.Dataset(val_x, label=val_y)
                    print(f"Feature data shape: {train_x.shape}")
                    print(f"Label data shape: {train_y.shape}")
                    print(f"Feature data type: {type(train_x)}")
                    print(f"Label data type: {type(train_y)}")

                    params = {
                        'task': 'train',
                        'boosting_type': 'gbdt',
                        'objective': 'binary',
                        'metric': {'binary_logloss'},
                        'num_leaves': 20,
                        'learning_rate': 0.01,
                        'feature_fraction': 0.9,
                        'bagging_fraction': 0.8,
                        'bagging_freq': 1,
                        'verbose': 0,
                        'is_unbalance': True
                    }
                    early_stopping_callback = early_stopping(stopping_rounds=100)
                    # train
                    model = lgb.train(params, train_matrix, valid_sets=[valid_matrix], callbacks=[early_stopping_callback],
                                      num_boost_round=200)

                    # lgb.plot_importance(model)
                    # plt.show()

                    # predict
                    test_pred = model.predict(test_x, num_iteration=model.best_iteration)
                    result_df = pd.DataFrame({
                        'id': test_id,
                        'true_label': test_y,
                        'predicted_prob': test_pred
                    })

                    result_df.to_csv(f'{result_dir}/result_{index}.csv', index=False)
                    print("Test results saved to 'result_{index}.csv'")

                    # save metrics
                    save_evaluation_metrics_to_arrays(test_y, test_pred, metric)
                    # clear_folder(embeddings_csv_dir)

                total_partitions = len(sorted_dataset_list)
                start_time = time_range.get(dataset_name)[0]
                end_time = time_range.get(dataset_name)[1]
                evaluation(result_dir, metric, dataset_name, model_name, delay, total_partitions, start_time, end_time)


if __name__ == "__main__":
    main()
