import pandas as pd
from node2vec import Node2Vec
import multiprocessing
import random
from data_process import load_graphs
random.seed(42)


def train_and_save_embeddings(embeddings_csv_dir, G, index, dataset_type, dimensions, walk_length, num_walks, workers):
    # 初始化Node2Vec模型
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=0.25, q=0.4,
                        workers=workers)
    # 训练模型
    model = node2vec.fit(window=4, min_count=1)

    # 获取节点嵌入
    embeddings = {node: model.wv[str(node)] for node in G.nodes()}

    # 将嵌入保存为CSV文件
    embeddings_df = pd.DataFrame.from_dict(embeddings, orient='index')
    embeddings_df.to_csv(f'{embeddings_csv_dir}/{index}_{dataset_type}_embeddings.csv')


def run_node2vec(dataset_dir, embeddings_csv_dir, index, dimensions, walk_length, num_walks, workers):
    Gx, Gy, Gz = load_graphs(dataset_dir, index)

    train_and_save_embeddings(embeddings_csv_dir, Gx, index, 'train', dimensions, walk_length, num_walks, workers)
    train_and_save_embeddings(embeddings_csv_dir, Gy, index, 'val', dimensions, walk_length, num_walks, workers)
    train_and_save_embeddings(embeddings_csv_dir, Gz, index, 'test', dimensions, walk_length, num_walks, workers)
