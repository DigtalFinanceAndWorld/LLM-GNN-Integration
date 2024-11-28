import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt

from single_expert.describe_single_node_testing import describe_single_node_test


def load_graph(file_path):
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def load_labels(file_path):
    with open(file_path, 'r') as f:
        labels = json.load(f)
    return labels


def visualize_node_transaction(graph, node, labels):
    """
    可视化某个节点的交易情况，包括入边、出边以及内部交易。
    :param graph: networkx图
    :param node: 要可视化的节点
    :param labels: 标签数据
    :return: None
    """
    # 创建一个子图，包含目标节点及其周围的交易
    subgraph_nodes = set([node])
    subgraph_edges = set()

    # 获取节点的所有入边和出边
    incoming_edges = list(graph.in_edges(node, data=True, keys=True))
    outgoing_edges = list(graph.out_edges(node, data=True, keys=True))
    print(incoming_edges)
    print(outgoing_edges)

    # 收集所有与目标节点相关的节点和边
    for edge in incoming_edges + outgoing_edges:
        subgraph_nodes.add(edge[0])  # 添加发起方
        subgraph_nodes.add(edge[1])  # 添加接收方
        subgraph_edges.add((edge[0], edge[1], edge[2]))

    # 构造子图
    subgraph = graph.subgraph(subgraph_nodes).copy()

    # 可视化布局
    pos = nx.spring_layout(subgraph)

    # 设置节点的颜色，根据是否为 phishing
    node_colors = ['red' if labels.get(n, 0) == 1 else 'green' for n in subgraph.nodes]

    # 绘制节点
    nx.draw_networkx_nodes(subgraph, pos, node_size=500, node_color=node_colors)

    # 绘制边
    nx.draw_networkx_edges(subgraph, pos, edgelist=subgraph_edges)

    # 绘制节点标签
    labels_mapping = {n: f"{n} ({'P' if labels.get(n, 0) == 1 else 'NP'})" for n in subgraph.nodes}
    nx.draw_networkx_labels(subgraph, pos, labels=labels_mapping)

    # 为每条边标注交易金额
    edge_labels = {}
    for edge in subgraph_edges:
        amount = graph.edges[edge].get('amount', 'N/A')
        edge_labels[(edge[0], edge[1])] = f"Amount: {amount}"

    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels)

    # 显示图形
    plt.title(f"Transaction Visualization for Node {node}")
    plt.show()


# 示例调用
if __name__ == '__main__':
    dataset_dir = f'../dataset/clustered_graphs/ZipZap/200/delay_5'
    graph_test_file_path = f'{dataset_dir}/3/test.pkl'
    labels_test_file_path = f'{dataset_dir}/3/test_labels.json'
    graph_test = load_graph(graph_test_file_path)
    labels_test = load_labels(labels_test_file_path)
    target_node = "merged_200"
    for i, node in enumerate(graph_test.nodes):
        print(node)
        if node == target_node:
            test_node_detail, is_phishing, _ = describe_single_node_test(graph_test, labels_test, node)
            visualize_node_transaction(graph_test, node, labels_test)
