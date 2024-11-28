import json
import pickle
import os
import networkx as nx
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
import scipy.sparse as sp
import torch


def load_graphs(dataset_dir, index):
    with open(f'{dataset_dir}/{index}/train.pkl', 'rb') as f:
        Gx = pickle.load(f)
    with open(f'{dataset_dir}/{index}/val.pkl', 'rb') as f:
        Gy = pickle.load(f)
    with open(f'{dataset_dir}/{index}/test.pkl', 'rb') as f:
        Gz = pickle.load(f)
    if not isinstance(Gx, nx.Graph):
        Gx = nx.Graph(Gx)
    if not isinstance(Gy, nx.Graph):
        Gy = nx.Graph(Gy)
    if not isinstance(Gz, nx.Graph):
        Gz = nx.Graph(Gz)

    return Gx, Gy, Gz


def read_csv_data(dataset_dir, embeddings_csv_dir, index):
    fs = pd.read_csv(f"{embeddings_csv_dir}/{str(index)}_train_embeddings.csv", header=0, index_col=0)
    print(len(fs))
    train_x = fs.values
    print(index)
    print(train_x.shape)
    with open(f'{dataset_dir}/{index}/train_labels.json', 'r', encoding='utf8') as f:
        lb = json.load(f)
    y = list(lb.values())
    train_y = np.array(y)

    fs1 = pd.read_csv(f"{embeddings_csv_dir}/{str(index)}_val_embeddings.csv", header=0, index_col=0)
    val_x = fs1.values
    with open(f'{dataset_dir}/{index}/val_labels.json', 'r', encoding='utf8') as f:
        lb1 = json.load(f)
    y1 = list(lb1.values())
    val_y = np.array(y1)

    fs2 = pd.read_csv(f"{embeddings_csv_dir}/{str(index)}_test_embeddings.csv", header=0, index_col=0)
    test_x = fs2.values
    with open(f'{dataset_dir}/{index}/test_labels.json', 'r', encoding='utf8') as f:
        lb2 = json.load(f)
    y2 = list(lb2.values())
    id = list(lb2.keys())
    test_y = np.array(y2)
    test_id = np.array(id)
    return train_x, train_y, val_x, val_y, test_x, test_y, test_id


def clear_folder(folder_path):
    # 确保目标文件夹存在
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # 检查是否是文件或符号链接，并删除
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                # 如果是目录，则递归删除目录中的所有内容
                elif os.path.isdir(file_path):
                    # 使用 os.walk 清空目录内容，然后删除目录本身
                    for root, dirs, files in os.walk(file_path, topdown=False):
                        # 删除文件
                        for name in files:
                            os.remove(os.path.join(root, name))
                        # 删除空目录
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    # 删除最外层目录
                    os.rmdir(file_path)
            except Exception as e:
                print(f"clear {file_path} error: {e}")
    else:
        print(f"{folder_path} not found！")


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, dataset_dir, dataset_type, transform=None, pre_transform=None):
        self.dataset_type = dataset_type
        self.dataset_dir = dataset_dir
        self.graph_path = f'{dataset_dir}/{dataset_type}.pkl'
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

    # 当第一次加载数据集时，Dataset类会检查处理后的文件是否存在。如果不存在，它会从原始文件开始处理数据，然后将结果保存为processed_file_names所指定的文件格式和名称。当下次加载数据集时，如果处理后的文件已存在，它就可以直接加载这些文件，而无需重新进行耗时的数据预处理。
    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        return [f'{self.dataset_type}_fs.json', f'{self.dataset_type}_labels.json']

    @property
    def processed_file_names(self):
        return [f'{self.dataset_type}_data.pt']

    @property
    def num_classes(self):
        return 2

    # 在自定义数据集类中，process 方法用于定义如何从原始数据文件（由 raw_file_names 指定）转换为处理后的数据（保存为 processed_file_names 指定的文件）。
    def process(self):
        # 检查是否已经处理过数据
        if os.path.exists(self.processed_paths[0]):
            return
        # 加载原始数据
        node_features_path = os.path.join(self.dataset_dir, f'{self.dataset_type}_fs.json')
        node_labels_path = os.path.join(self.dataset_dir, f'{self.dataset_type}_labels.json')

        with open(node_features_path, 'r') as file:
            node_features = json.load(file)

        with open(node_labels_path, 'r') as file:
            node_labels = json.load(file)

        labels = list(node_labels.values())
        labels_tensor = torch.tensor(labels, dtype=torch.float)

        f = open(self.graph_path, 'rb')
        G1 = pickle.load(f)

        node_to_index = {node: idx for idx, node in enumerate(node_features.keys())}

        # 初始化特征矩阵和标签数组
        num_nodes = len(node_features)
        num_features = len(next(iter(node_features.values())))
        features = np.zeros((num_nodes, num_features))

        for node, fs in node_features.items():
            node_index = node_to_index[node]
            features[node_index, :] = fs

        node_feature = features

        adj_matrix = self.build_undirected_adjacency_matrix(G1)
        norm_adj_matrix = self.normalize_adjacency_matrix(adj_matrix)

        source_nodes = []
        target_nodes = []

        for node1, node2 in G1.edges():
            u = node_to_index[node1]
            v = node_to_index[node2]
            source_nodes.append(u)
            target_nodes.append(v)

        # 转换为张量
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        # 将 NumPy 数组转换为 PyTorch 张量
        features_tensor = torch.FloatTensor(node_feature)
        indices = torch.from_numpy(np.vstack((norm_adj_matrix.row, norm_adj_matrix.col))).long()
        values = torch.FloatTensor(norm_adj_matrix.data)
        tensor_adjacency = torch.sparse_coo_tensor(indices, values, size=torch.Size(norm_adj_matrix.shape))

        # 构建 PyTorch Geometric Data 对象
        data = Data(x=features_tensor, y=labels_tensor, edge_index=edge_index, adj_matrix=tensor_adjacency)
        data_list = []
        data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save(data, self.processed_paths[0])

    def __len__(self):
        return len(self.processed_file_names)

    # def len(self):
    #     return 1

    @property
    def processed_paths(self):
        return [os.path.join(self.processed_dir, f'{self.dataset_type}_data.pt')]

    def get(self, idx):
        if idx != 0:
            raise IndexError('Index out of range')
        data = torch.load(os.path.join(self.processed_dir, f'{self.dataset_type}_data.pt'))
        return data

    def build_undirected_adjacency_matrix(self, G):
        if G.number_of_edges() == 0:
            print('xxxxxx')
        else:
            print('yyyyyy')
        # 创建节点到索引的映射
        node_to_int = {node: i for i, node in enumerate(G.nodes())}
        num_nodes = len(node_to_int)

        # 初始化行、列和数据列表
        row, col, data = [], [], []

        # 遍历每条边，添加无向连接
        for u, v in G.edges():
            row.append(node_to_int[u])
            col.append(node_to_int[v])
            data.append(1)

            # 因为是无向图，所以需要添加反向边
            if u != v:  # 防止自环被加倍计数
                row.append(node_to_int[v])
                col.append(node_to_int[u])
                data.append(1)

        # 创建稀疏矩阵，合并多重边的权重
        adj_matrix = sp.coo_matrix((data, (row, col)), shape=(G.number_of_nodes(), G.number_of_nodes()),
                                   dtype=np.float32)
        adj_matrix.sum_duplicates()  # 合并多重边的权重
        return adj_matrix

    def normalize_adjacency_matrix(self, adj):
        # 将邻接矩阵转换为COO格式（如果尚未转换）
        adj = sp.coo_matrix(adj)
        # 添加自环
        adj = adj + sp.eye(adj.shape[0])
        # 计算度矩阵D
        rowsum = np.array(adj.sum(1))
        # 计算D^{-1/2}
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # 计算归一化的邻接矩阵D^{-1/2} * A * D^{-1/2}
        normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

        return normalized_adj


def load_gnn_data(dataset_dir, embeddings_csv_dir, index, device):
    train_dataset = MyOwnDataset(embeddings_csv_dir, f'{dataset_dir}/{index}', "train")
    train_data = train_dataset[0]
    train_data.x = train_data.x.to(device)
    train_data.x = F.normalize(train_data.x)
    train_data.edge_index = train_data.edge_index.to(device)

    val_dataset = MyOwnDataset(embeddings_csv_dir, f'{dataset_dir}/{index}', "val")
    val_data = val_dataset[0]
    val_data.x = val_data.x.to(device)
    val_data.x = F.normalize(val_data.x)
    val_data.edge_index = val_data.edge_index.to(device)

    test_dataset = MyOwnDataset(embeddings_csv_dir, f'{dataset_dir}/{index}', "test")
    test_data = test_dataset[0]
    test_data.x = test_data.x.to(device)
    test_data.x = F.normalize(test_data.x)
    test_data.edge_index = test_data.edge_index.to(device)
    return train_data, val_data, test_data
