import json
import pickle
from describe_single_node_testing import describe_single_node_train


def load_graph(file_path):
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def load_json(file_path):
    with open(file_path, 'r') as f:
        labels = json.load(f)
    return labels


def load_nodemap(file_path):
    with open(file_path, 'r') as f:
        nodemap = json.load(f)
    return nodemap


def generate_train_node_detail(graph, labels):

    # 筛选符合条件的节点
    phishing_nodes = [node for node in graph.nodes if labels.get(node, 0) == 1]
    non_phishing_nodes = [node for node in graph.nodes if labels.get(node, 0) == 0]
    # 生成自然语言描述
    node_descriptions = {}
    # 限制数量：取前5个
    min_description_length = 5000
    max_description_length = 30000
    phishing_samples = []
    non_phishing_samples = []

    n = 3
    # 描述每个钓鱼节点，限制描述长度
    for node in phishing_nodes:
        description = describe_single_node_train(graph, labels, node)
        if min_description_length <= len(description) <= max_description_length:
            print("add train example: phishing_nodes ...")
            phishing_samples.append(node)
            node_descriptions[node] = description
        if len(phishing_samples) >= n:
            break

    # 描述每个非钓鱼节点，限制描述长度
    for node in non_phishing_nodes:
        description = describe_single_node_train(graph, labels, node)
        if min_description_length <= len(description) <= max_description_length:
            print("add train example: non_phishing_nodes ...")
            non_phishing_samples.append(node)
            node_descriptions[node] = description
        if len(non_phishing_samples) >= n:
            break

    # 拼接自然语言描述
    new_node_detail = "\n"
    new_node_detail += "----- Phishing Nodes -----\n"
    for node in phishing_samples:
        new_node_detail += node_descriptions[node]
    new_node_detail += "----- End of Phishing Nodes -----\n\n"
    new_node_detail += "----- Not Phishing Nodes -----\n"
    for node in non_phishing_samples:
        new_node_detail += node_descriptions[node] + "\n"
    new_node_detail += "----- End of Not Phishing Nodes -----\n"

    return new_node_detail
