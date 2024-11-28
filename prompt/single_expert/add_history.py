from datetime import timedelta
import transformers
from describe_single_node_testing_token_4096 import *


# 缩略后的节点作为新的节点，加入所有历史节点，成为一张大图，这样就可以获取历史
def add_history(graph_all, graph, labels, nodemap, start_date_str, delay):
    graph = graph.copy()
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    print(f"[INFO] 已将 start_date 字符串转换为 datetime 对象：{start_date}")

    # 计算历史结束时间戳
    history_end = start_date - timedelta(days=delay)
    history_end_timestamp = int(history_end.timestamp())
    print(f"[INFO] 历史结束时间戳（A）：{history_end_timestamp}，对应日期：{history_end}")

    # 创建合并节点名称到原始节点集合的映射
    merged_node_to_original_nodes = {}  # key: 合并节点名称, value: 原始节点地址集合
    merged_num_to_node_name = {}  # key: 合并节点编号, value: 合并节点名称

    # 初始化合并节点及其原始节点
    print("[INFO] 初始化合并节点及其原始节点...")
    for node in graph.nodes():
        if node.startswith('merged_'):
            merged_node_num = int(node[len('merged_'):])
            merged_num_to_node_name[merged_node_num] = node
            merged_node_to_original_nodes[node] = set()
            # print(f"[DEBUG] 发现合并节点：{node}，编号：{merged_node_num}")
        else:
            # 如果节点不是合并节点，打印警告
            print(f"[WARNING] 节点 {node} 不是合并节点，可能需要检查")

    # 使用 nodemap 填充 merged_node_to_original_nodes
    print("[INFO] 使用 nodemap 填充合并节点的原始节点集合...")
    for address, merged_num in nodemap.items():
        merged_node_name = merged_num_to_node_name.get(merged_num)
        if merged_node_name:
            merged_node_to_original_nodes[merged_node_name].add(address)
        else:
            print(f"[WARNING] 在 nodemap 中找不到合并节点编号 {merged_num}，对应地址 {address}")

    # 收集缩略图中包含的所有原始节点
    all_original_nodes_in_graph = set()
    print("[INFO] 收集所有原始节点...")
    for merged_node, original_nodes in merged_node_to_original_nodes.items():
        all_original_nodes_in_graph.update(original_nodes)
        # print(f"[DEBUG] 合并节点 {merged_node} 包含 {len(original_nodes)} 个原始节点")

    # 准备收集新节点和边
    new_nodes = set()

    # 遍历 graph_all 中的所有边
    print("[INFO] 遍历 graph_all 中的所有边...")
    edge_count = 0
    for u, v, key, data in graph_all.edges(data=True, keys=True):
        timestamp = data.get('timestamp', 0)
        if timestamp >= history_end_timestamp:
            continue  # 跳过发生在 A 之后的边

        # 检查任一节点是否在缩略图的原始节点中
        u_in_graph = u in all_original_nodes_in_graph
        v_in_graph = v in all_original_nodes_in_graph

        if u_in_graph or v_in_graph:
            edge_count += 1
            if u_in_graph and v_in_graph:
                # 两个节点都在合并节点中
                u_merged_node = next(
                    (mn for mn, nodes in merged_node_to_original_nodes.items() if u in nodes), None)
                v_merged_node = next(
                    (mn for mn, nodes in merged_node_to_original_nodes.items() if v in nodes), None)

                if u_merged_node == v_merged_node:
                    # 两个节点在同一个合并节点中
                    internal_edges = graph.nodes[u_merged_node].get('internal_edges', [])
                    internal_edges.append((u, v, key, data))
                    graph.nodes[u_merged_node]['internal_edges'] = internal_edges
                    # print(f"[DEBUG] 添加内部边：{u} -> {v} 到合并节点 {u_merged_node}")
                else:
                    # 边连接两个不同的合并节点
                    graph.add_edge(u_merged_node, v_merged_node, key=key, **data)
                    # print(f"[DEBUG] 添加边：{u_merged_node} -> {v_merged_node}")
            else:
                # 一个节点在合并节点中，另一个是新节点
                if u_in_graph:
                    # u 在合并节点中，v 是新节点
                    new_nodes.add(v)
                    u_merged_node = next(
                        (mn for mn, nodes in merged_node_to_original_nodes.items() if u in nodes), None)
                    graph.add_edge(u_merged_node, v, key=key, **data)
                    # print(f"[DEBUG] 添加边：{u_merged_node} -> 新节点 {v}")
                else:
                    # v 在合并节点中，u 是新节点
                    new_nodes.add(u)
                    v_merged_node = next(
                        (mn for mn, nodes in merged_node_to_original_nodes.items() if v in nodes), None)
                    graph.add_edge(u, v_merged_node, key=key, **data)
                    # print(f"[DEBUG] 添加边：新节点 {u} -> {v_merged_node}")

    print(f"[INFO] 总共处理了 {edge_count} 条相关边")
    print(f"[INFO] 新增节点数量：{len(new_nodes)}")

    # 将新节点添加到图中，并更新 test_label
    # print("[INFO] 添加新节点并更新标签...")
    for node in new_nodes:
        if graph_all.has_node(node):
            node_attrs = graph_all.nodes[node]
            graph.add_node(node, **node_attrs)

            # 从节点属性中获取 'isp' 值
            isp_value = node_attrs.get('isp', 0)
            label = 1 if isp_value == 1 else 0
            labels[node] = label
        else:
            graph.add_node(node)
            # 如果节点没有属性，无法确定标签，默认为 0
            labels[node] = 0

    return graph, labels


def single_expert_model(test_node_detail):
    messages = f"""
### Instruction  
You are a blockchain analyst specializing in identifying fraudulent behavior within the Ethereum network. Your task is to analyze the transaction data of a **Target Node** and classify it as either a **Phishing Node** or **Non-Phishing Node**. This is a **binary classification problem**, requiring you to return one of the following:  
- **1**: Phishing Node  
- **0**: Non-Phishing Node  

### Task Description  
Evaluate the Target Node’s behavior based on these key attributes:  
1. **Node Degree**: The number of incoming and outgoing transactions. Suspicious nodes may exhibit unusually high outgoing transactions with low or zero-value amounts.  
2. **Transaction Amounts**: Look for unusual average transaction amounts, both incoming and outgoing, including internal transactions.  
3. **Transaction Frequency**: Check for high frequencies or sudden bursts of transactions, which could indicate malicious activity.  
4. **Counterparty Interaction**: Identify frequent interactions with unknown nodes, suspicious addresses, or a large number of zero-value transactions.  
5. **Internal Transactions**: Look for frequent internal transfers with no clear financial purpose, which could signal deceptive behavior.  
6. **Transaction Timeline**: Determine whether transactions are spread evenly over time or clustered in bursts, which might indicate phishing.  

### Objective  
Accurately identifying phishing nodes is critical to protecting the financial ecosystem. Use the provided data to predict whether the node exhibits phishing behavior. Your output must be a precise label:  
- **1**: Phishing Node
- **0**: Non-Phishing Node

### Target Node Information  
{test_node_detail}

### Output Label

"""
    return messages


if __name__ == '__main__':
    tokenizer = transformers.AutoTokenizer.from_pretrained("/home/a/zmb_workspace/model/Llama-3.1-8B")
    dataset_name = "MulDiGraph"
    multiple = "500"
    delay = 5

    graph_nodes = 0
    index_2048 = 0
    index_3072 = 0
    index_4096 = 0
    for index in range(5, 10):
        index = str(index)
        dataset_dir = f'../../dataset/clustered_graphs/{dataset_name}/{multiple}/delay_{delay}'
        start_date_file_path = f'../../dataset/{dataset_name}/start_times.json'
        # train_start_date = load_json(start_date_file_path)["train"][index]
        test_start_date = load_json(start_date_file_path)["test"][index]

        graph_all_file_path = f'../../dataset/{dataset_name}/data/{dataset_name}.pkl'
        graph_file_path = f'{dataset_dir}/{index}/test.pkl'
        labels_file_path = f'{dataset_dir}/{index}/test_labels.json'
        nodemap_file_path = f'{dataset_dir}/{index}/test_nodemap.json'
        graph_all = load_graph(graph_all_file_path)
        graph = load_graph(graph_file_path)
        labels = load_json(labels_file_path)
        nodemap = load_json(nodemap_file_path)

        add_histroy_graph, labels_test = add_history(graph_all, graph, labels, nodemap, test_start_date, delay)

        graph_nodes += len(graph)
        for i, node in enumerate(graph.nodes):
            test_node_detail, is_phishing = describe_single_node_test(add_histroy_graph, labels_test, node)
            messages = single_expert_model(test_node_detail)
            test_node_detail_token = len(tokenizer.encode(test_node_detail, add_special_tokens=False))
            messages_token = len(tokenizer.encode(messages, add_special_tokens=False))
            print(f"================ Node {i} ================")
            print(f"test_node_detail count_tokens: {len(tokenizer.encode(test_node_detail, add_special_tokens=False))}")
            print(f"messages count_tokens: {len(tokenizer.encode(messages, add_special_tokens=False))}")
            print(f"is_phishing: {is_phishing}")
            print(test_node_detail)

            if messages_token > 2048:
                index_2048 += 1
                print("messages_token > 2048")
                if messages_token > 3072:
                    index_3072 += 1
                    print("messages_token > 3072")
                    if messages_token > 4096:
                        index_4096 += 1
                        print("messages_token > 4096")


    print(f"graph nodes: {graph_nodes}")
    print(f"token > 2048: {index_2048}")
    print(f"token > 4096: {index_4096}")

