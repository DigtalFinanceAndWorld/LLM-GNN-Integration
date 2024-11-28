import pymetis
import pickle
import networkx as nx
import numpy as np
import copy
from tqdm import tqdm
import os
import json
from multiprocessing import Pool, Manager, cpu_count
from time import sleep


def merge_nodes_with_same_label(graph, labels):
    # 检查是否是MultiDiGraph类型
    if not isinstance(graph, nx.MultiDiGraph):
        raise ValueError("This function is designed to work with MultiDiGraph structures only.")

    # 深拷贝原始图，防止修改时影响原图
    origin_graph = copy.deepcopy(graph)

    # 遍历所有标签
    for label in set(labels):
        # 找到具有相同 label 的所有节点
        nodes = np.array(origin_graph.nodes())[np.where(np.array(labels) == label)]
        merged_node = 'merged_' + str(label)

        # 添加合并节点到图中
        graph.add_node(merged_node)

        # 内部边集合，用于记录合并节点之间的多重边
        internal_edges = []

        # 反转图，用于获取输入边（MultiDiGraph 支持反转操作）
        reversed_graph = nx.reverse(graph, copy=True)

        # 合并节点
        for node in nodes:
            if node not in graph:
                # 检查节点是否已经被删除，避免 KeyError
                continue

            # 获取节点的所有输出邻居
            out_neighbors = list(graph.successors(node))
            # 获取节点的所有输入邻居（通过反转图）
            in_neighbors = list(reversed_graph.successors(node))

            # 遍历所有输出边，合并到新节点
            for neighbor in out_neighbors:
                if neighbor not in graph:
                    # 检查邻居是否存在于图中，避免 KeyError
                    continue
                if neighbor not in nodes:
                    # 遍历所有多重边属性（对于MultiDiGraph，需要遍历每个键）
                    for key, edge_data in graph.get_edge_data(node, neighbor).items():
                        # 添加每一条多重边到合并后的节点（保留所有属性）
                        graph.add_edge(merged_node, neighbor, key=key, **edge_data)
                else:
                    # 如果是内部边，添加到内部边集合中
                    for key, edge_data in graph.get_edge_data(node, neighbor).items():
                        internal_edges.append((node, neighbor, key, edge_data))

            # 遍历所有输入边，合并到新节点
            for neighbor in in_neighbors:
                if neighbor not in graph:
                    continue
                if neighbor not in nodes:
                    # 遍历所有多重边属性（对于MultiDiGraph，需要遍历每个键）
                    for key, edge_data in graph.get_edge_data(neighbor, node).items():
                        # 添加每一条多重边到合并后的节点
                        graph.add_edge(neighbor, merged_node, key=key, **edge_data)
                else:
                    # 如果是内部边，添加到内部边集合中
                    for key, edge_data in graph.get_edge_data(neighbor, node).items():
                        internal_edges.append((neighbor, node, key, edge_data))

            # 删除当前节点
            if node in graph:
                graph.remove_node(node)

        # 处理内部边，将其作为合并节点的自环边（可选）或附加到合并节点属性中
        for (node1, node2, key, edge_data) in internal_edges:
            # 方法1：将内部边信息保留为合并节点的自环边（可选）
            graph.add_edge(merged_node, merged_node, key=key, **edge_data)

            # 方法2：将内部边记录为合并节点的一个属性，存储为内部边列表（可选）
            if 'internal_edges' not in graph.nodes[merged_node]:
                graph.nodes[merged_node]['internal_edges'] = []
            graph.nodes[merged_node]['internal_edges'].append((node1, node2, key, edge_data))

    return graph


def run_graph_clustering(dataset, delay, i, type):
    multiple = 100
    file_path = f'/home/a/zmb_workspace/product/Phisher_detect/dataset/{dataset}/data/LLM//delay_{delay}/{i}/{type}.pkl'
    label_path = f'/home/a/zmb_workspace/product/Phisher_detect/dataset/{dataset}/data/LLM//delay_{delay}/{i}/{type}_labels.json'
    output_graph_path = f'/home/a/zmb_workspace/product/Phisher_detect/dataset/clustered_graphs/{dataset}/{multiple}/delay_{delay}/{i}/{type}.pkl'
    output_label_path = f'/home/a/zmb_workspace/product/Phisher_detect/dataset/clustered_graphs/{dataset}/{multiple}/delay_{delay}/{i}/{type}_labels.json'
    output_map_path = f'/home/a/zmb_workspace/product/Phisher_detect/dataset/clustered_graphs/{dataset}/{multiple}/delay_{delay}/{i}/{type}_nodemap.json'

    # 检查输出路径是否存在，不在创建
    if not os.path.exists(os.path.dirname(output_graph_path)):
        os.makedirs(os.path.dirname(output_graph_path))
    # print(f'Clustering begins! Path: {file_path}')
    with open(file_path, 'rb') as f:
        G = pickle.load(f)
    with open(label_path, 'rb') as f:
        origin_labels = json.load(f)

    UG = nx.to_undirected(G)
    node_label_mapping = {node: i for i, node in enumerate(UG.nodes())}
    UG_numeric = nx.relabel_nodes(UG, node_label_mapping)
    adjacency_list = list(nx.to_dict_of_lists(UG_numeric).values())
    n_cuts, labels = pymetis.part_graph(nparts=(len(G) // multiple), adjacency=adjacency_list)
    label_dict = {}
    node_map = dict(zip(list(G.nodes()), labels))
    for label in set(labels):
        label_name = f'merged_{label}'
        label_dict[label_name] = 0
        nodes = np.array(G.nodes())[np.where(np.array(labels) == label)]
        for node in nodes:
            if origin_labels[node] == 1:
                label_dict[label_name] = 1
                break

    zero_count = sum(1 for v in label_dict.values() if v == 0)
    one_count = sum(1 for v in label_dict.values() if v == 1)
    total_count = zero_count + one_count
    one_ratio_percent = (one_count / total_count * 100) if total_count > 0 else 0

    print(f"file_path: {file_path}")
    print(f"Number of 0: {zero_count}")
    print(f"Number of 1: {one_count}")
    print(f"Proportion of 1: {one_ratio_percent:.2f}%")
    print("================================================")
    with open(output_map_path, 'w') as f:
        json.dump(node_map, f)
    with open(output_label_path, 'w') as f:
        json.dump(label_dict, f)
    # print(f'Labeling complete! Path: {output_label_path}')
    graph = merge_nodes_with_same_label(G, labels)
    with open(output_graph_path, 'wb') as f:
        pickle.dump(graph, f)
    print(f'Clustering complete! Path: {output_graph_path}')
    return


if __name__ == '__main__':
    pool = Pool(8)
    delay_list = ['5']
    pool_args = [('MulDiGraph', delay, str(i), 'train') for delay in delay_list for i in range(1, 19)]
    pool_args += [('MulDiGraph', delay, str(i), 'test') for delay in delay_list for i in range(1, 19)]
    # pool_args = [('ZipZap', delay, str(i), 'train') for delay in delay_list for i in range(12, 51)]
    # pool_args += [('ZipZap', delay, str(i), 'test') for delay in delay_list for i in range(12, 51)]
    pool.starmap(run_graph_clustering, pool_args)
    pool.close()
    pool.join()
