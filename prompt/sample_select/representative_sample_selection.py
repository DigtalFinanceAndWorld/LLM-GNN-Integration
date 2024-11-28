import json
import pickle
import networkx as nx
import pandas as pd

from prompt.single_expert.describe_single_node_testing import describe_single_node_train


def load_graph(file_path):
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def load_json(file_path):
    with open(file_path, 'r') as f:
        labels = json.load(f)
    return labels


def generate_representative_details(subgraph_file_path_list, labels_file_path_list, top_n_positive, top_n_negative):
    merged_graph = merge_multiple_subgraphs(subgraph_file_path_list)
    labels_dict = merge_json_list(labels_file_path_list)
    return most_representative_sample_selection(merged_graph, labels_dict, top_n_positive, top_n_negative)


def representative_sample_selection(graph, labels, top_n_positive=10, top_n_negative=10):

    # 1. 计算图结构特征（适用于 MultiDiGraph 的特征）
    degree_centrality = nx.out_degree_centrality(graph)  # 计算节点的出度中心性
    betweenness_centrality = nx.betweenness_centrality(graph, weight='weight')  # 使用边权重计算中介中心性
    pagerank_scores = nx.pagerank(graph, weight='weight')  # 使用边权重计算 PageRank 分数

    # 2. 自定义特征计算：计算每个节点的多重边权重总和
    multi_edge_weight_sum = {node: sum(data.get('weight', 1) for _, _, data in graph.edges(node, data=True))
                             for node in graph.nodes}

    # 3. 将多种特征融合到一个 DataFrame 中
    nodes_data = pd.DataFrame({
        'node': list(graph.nodes),
        'degree_centrality': [degree_centrality[node] for node in graph.nodes],
        'betweenness_centrality': [betweenness_centrality[node] for node in graph.nodes],
        'pagerank_scores': [pagerank_scores[node] for node in graph.nodes],
        'multi_edge_weight_sum': [multi_edge_weight_sum[node] for node in graph.nodes],
        'label': [labels.get(node, 0) for node in graph.nodes]
    })

    # 4. 标准化各项特征
    nodes_data[['degree_centrality', 'betweenness_centrality', 'pagerank_scores', 'multi_edge_weight_sum']] = \
        nodes_data[['degree_centrality', 'betweenness_centrality', 'pagerank_scores', 'multi_edge_weight_sum']].apply(
            lambda x: (x - x.mean()) / x.std())

    # 5. 计算综合得分（可以简单加权平均）
    nodes_data['composite_score'] = nodes_data[
        ['degree_centrality', 'betweenness_centrality', 'pagerank_scores', 'multi_edge_weight_sum']].mean(axis=1)

    # 6. 按标签分组，并从每个组中选择 top_n 节点
    positive_samples = nodes_data[nodes_data['label'] == 1].nlargest(top_n_positive, 'composite_score')
    negative_samples = nodes_data[nodes_data['label'] == 0].nlargest(top_n_negative, 'composite_score')

    # 7. 生成自然语言描述
    node_descriptions = {}

    # 描述每个阳性（钓鱼）节点
    for node in positive_samples['node']:
        node_descriptions[node] = describe_single_node_train(graph, labels, node)

    # 描述每个阴性（非钓鱼）节点
    for node in negative_samples['node']:
        node_descriptions[node] = describe_single_node_train(graph, labels, node)

    # 8. 拼接自然语言描述
    new_node_detail = "\n"
    new_node_detail += "----- Phishing Nodes -----\n"
    for node in positive_samples['node']:
        new_node_detail += node_descriptions[node] + "\n"
    new_node_detail += "----- End of Phishing Nodes -----\n"
    new_node_detail += "----- Not Phishing Nodes -----\n"
    for node in negative_samples['node']:
        new_node_detail += node_descriptions[node] + "\n"
    new_node_detail += "----- End of Not Phishing Nodes -----\n"

    return new_node_detail


# 将多个 nx.MultiDiGraph 子图合并为一个整体图
def merge_multiple_subgraphs(subgraph_file_path_list):
    # 创建一个空的 MultiDiGraph 作为合并后的整体图
    merged_graph = nx.MultiDiGraph()

    # 遍历所有子图，将它们合并到 merged_graph 中
    for subgraph_file_path in subgraph_file_path_list:
        subgraph = load_graph(subgraph_file_path)
        # 使用合并图的方式，将子图的节点和边添加到 merged_graph 中
        merged_graph = nx.compose(merged_graph, subgraph)

    return merged_graph


# 将包含多个 JSON 字典的列表进行融合，去除重复键，并合并值
def merge_json_list(labels_file_path_list):
    merged_dict = {}  # 存储合并后的字典

    # 逐个遍历 JSON 字典，并将其内容合并到 merged_dict 中
    for labels_file_path in labels_file_path_list:
        data = load_json(labels_file_path)
        for key, value in data.items():
            if key in merged_dict:
                continue
            else:
                merged_dict[key] = value

    return merged_dict


if __name__ == '__main__':
    index = 5
    dataset_dir = '../../dataset/clustered_graphs/MulDiGraph/1000/delay_5'

    subgraph_file_path_list = []
    labels_file_path_list = []
    for i in range(1, int(index) + 1):
        graph_train_file_path = f'{dataset_dir}/{str(i)}/train.pkl'
        labels_train_file_path = f'{dataset_dir}/{str(i)}/train_labels.json'
        subgraph_file_path_list.append(graph_train_file_path)
        labels_file_path_list.append(labels_train_file_path)
    train_node_detail = generate_representative_details(subgraph_file_path_list, labels_file_path_list)
    print(train_node_detail)
