import numpy as np
from gensim.models import Word2Vec
import tqdm
import random
import pandas as pd
from data_process import load_graphs
random.seed(42)


# 保存嵌入
def save_embeddings_to_csv(G, model, file_name):
    embeddings = {}
    for node in G.nodes():
        embeddings[node] = model.wv[str(node)]

    # 将嵌入保存为csv文件
    embeddings_df = pd.DataFrame.from_dict(embeddings, orient='index')
    embeddings_df.to_csv(file_name)


# 生成随机游走 修改本代码。使用双向队列
def generate_random_walks(G, num_walks, walk_length):
    walks = []
    for _ in tqdm.trange(num_walks, desc="Generating Walks"):
        nodes = list(G.nodes())
        np.random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                cur = walk[-1]
                neighbors = list(G.neighbors(cur))
                if not neighbors:
                    break
                walk.append(np.random.choice(neighbors))
            walks.append(walk)
    return walks


# DeepWalk算法
def deepwalk(embeddings_csv_dir, G, index, dataset_type, dimensions, walk_length, num_walks, window_size, workers):
    walks = generate_random_walks(G, num_walks, walk_length)
    model = Word2Vec(walks, vector_size=dimensions, window=window_size, min_count=1, sg=1, workers=workers)
    save_embeddings_to_csv(G, model, f'{embeddings_csv_dir}/{index}_{dataset_type}_embeddings.csv')


def run_deepwalk(dataset_dir, embeddings_csv_dir, index, dimensions, walk_length, num_walks,
                 window_size, workers):
    Gx, Gy, Gz = load_graphs(dataset_dir, index)

    deepwalk(embeddings_csv_dir, Gx, index, 'train', dimensions, walk_length, num_walks, window_size, workers)
    deepwalk(embeddings_csv_dir, Gy, index, 'val', dimensions, walk_length, num_walks, window_size, workers)
    deepwalk(embeddings_csv_dir, Gz, index, 'test', dimensions, walk_length, num_walks, window_size, workers)
