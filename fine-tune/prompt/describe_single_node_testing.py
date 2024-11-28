import json
import pickle
from datetime import datetime
from collections import defaultdict


def load_graph(file_path):
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def load_json(file_path):
    with open(file_path, 'r') as f:
        labels = json.load(f)
    return labels


def describe_single_node_train(graph, labels, node):
    # 获取节点与索引的映射关系以及节点度中心性
    index_mapping = create_index_mapping(graph)  # 假设该函数返回节点与索引的映射
    degree_centrality = calculate_degree_centrality(graph)  # 假设该函数返回度中心性

    # 节点的基本信息
    is_phishing = labels.get(node, 0)  # 获取合并节点的标签信息
    index = index_mapping.get(node, None)  # 使用 get 方法获取，避免 node 不存在时抛出 KeyError
    index_str = str(index) if index is not None else "N/A"

    # 获取节点的度数信息
    in_degree = graph.in_degree(node) if graph.in_degree(node) is not None else 0
    out_degree = graph.out_degree(node) if graph.out_degree(node) is not None else 0
    total_external_transactions = in_degree + out_degree

    # 获取内部交易边（根据合并节点属性 'internal_edges'）
    internal_edges = graph.nodes[node].get('internal_edges', [])

    # 获取合并节点中的内部节点集合，用于区分内部和外部交易
    internal_nodes = set([node1 for node1, _, _, _ in internal_edges] + [node2 for _, node2, _, _ in internal_edges])

    # 收集所有入边和出边的详细信息
    incoming_edges = [(sender, node, key, data) for sender, _, key, data in graph.in_edges(node, data=True, keys=True)]
    outgoing_edges = [(node, recipient, key, data) for _, recipient, key, data in
                      graph.out_edges(node, data=True, keys=True)]

    # 区分内部交易和外部交易（使用内部节点集合判断）
    internal_edges_combined = [edge for edge in incoming_edges + outgoing_edges
                               if edge[0] == node and edge[1] == node]
    total_internal_transactions = len(internal_edges_combined)

    # 外部交易：发起方或接收方不是内部节点（且不是待测节点自身）
    external_incoming_edges = [edge for edge in incoming_edges if edge[0] not in internal_nodes and edge[0] != node]
    external_outgoing_edges = [edge for edge in outgoing_edges if edge[1] not in internal_nodes and edge[1] != node]

    # 计算内部交易的平均交易金额（所有内部交易合并到一起计算）
    total_internal_amount = sum(edge_data.get('amount', 0) for _, _, _, edge_data in internal_edges_combined)
    avg_internal_amount = total_internal_amount / len(internal_edges_combined) if len(
        internal_edges_combined) > 0 else 0

    # 计算外部交易的平均交易金额
    avg_external_incoming_amount = (
            sum(edge_data.get('amount', 0) for _, _, _, edge_data in external_incoming_edges) / len(
        external_incoming_edges)) if len(external_incoming_edges) > 0 else 0
    avg_external_outgoing_amount = (
            sum(edge_data.get('amount', 0) for _, _, _, edge_data in external_outgoing_edges) / len(
        external_outgoing_edges)) if len(external_outgoing_edges) > 0 else 0

    # 计算交易时间
    transaction_timestamps = [
        edge_data['timestamp'] for _, _, _, edge_data in incoming_edges + outgoing_edges if 'timestamp' in edge_data
    ]

    first_transaction_time = min(transaction_timestamps, default=None)
    last_transaction_time = max(transaction_timestamps, default=None)

    first_transaction_time_str = convert_timestamp(first_transaction_time) if first_transaction_time else 'N/A'
    last_transaction_time_str = convert_timestamp(last_transaction_time) if last_transaction_time else 'N/A'

    # 计算交易频率
    total_transactions = len(incoming_edges) + len(outgoing_edges)  # 总交易数（外部）
    if first_transaction_time and last_transaction_time and first_transaction_time != last_transaction_time:
        total_time_span_seconds = last_transaction_time - first_transaction_time
        total_time_span_days = total_time_span_seconds / 86400  # 将秒转换为天
        avg_transaction_frequency = total_transactions / total_time_span_days if total_time_span_days > 0 else 0
    else:
        avg_transaction_frequency = 0  # 如果没有有效的时间跨度，频率设为 0

    # 节点的度中心性
    centrality = degree_centrality.get(node, 0)  # 如果 centrality 不存在，使用 0 作为默认值

    # 外部交易的汇总信息
    external_incoming_summary = defaultdict(lambda: {"count": 0, "total_amount": 0, "last_timestamp": 0})
    external_outgoing_summary = defaultdict(lambda: {"count": 0, "total_amount": 0, "last_timestamp": 0})

    # 处理外部入边
    for sender, _, key, data in external_incoming_edges:
        sender_index = index_mapping.get(sender, "N/A")
        if sender.startswith('merged_'):
            sender_label = "Unknown"
        else:
            sender_label = "Phishing" if labels.get(sender, 0) == 1 else "Non-Phishing"
        amount = data.get('amount', 0)
        timestamp = data.get('timestamp', 0)
        # 汇总该节点的交易信息
        external_incoming_summary[(sender_index, sender_label)]["count"] += 1
        external_incoming_summary[(sender_index, sender_label)]["total_amount"] += amount
        external_incoming_summary[(sender_index, sender_label)]["last_timestamp"] = max(
            external_incoming_summary[(sender_index, sender_label)]["last_timestamp"], timestamp
        )

    # 处理外部出边
    for _, recipient, key, data in external_outgoing_edges:
        recipient_index = index_mapping.get(recipient, "N/A")
        if recipient.startswith('merged_'):
            recipient_label = "Unknown"
        else:
            recipient_label = "Phishing" if labels.get(recipient, 0) == 1 else "Non-Phishing"
        amount = data.get('amount', 0)
        timestamp = data.get('timestamp', 0)
        # 汇总该节点的交易信息
        external_outgoing_summary[(recipient_index, recipient_label)]["count"] += 1
        external_outgoing_summary[(recipient_index, recipient_label)]["total_amount"] += amount
        external_outgoing_summary[(recipient_index, recipient_label)]["last_timestamp"] = max(
            external_outgoing_summary[(recipient_index, recipient_label)]["last_timestamp"], timestamp
        )

    internal_transactions_summary = defaultdict(int)
    for sender, recipient, key, data in internal_edges_combined:
        sender_index = index_mapping.get(sender, "N/A")
        recipient_index = index_mapping.get(recipient, "N/A")
        amount = data.get('amount', 'N/A')
        transaction_key = (sender_index, recipient_index, amount)
        internal_transactions_summary[transaction_key] += 1

    transaction_details = "**External Incoming Transactions**\n"
    # 用于汇总所有 "Non-Phishing" 节点的数据
    non_phishing_count = 0  # 记录 "Non-Phishing" 节点的数量
    non_phishing_total_transactions = 0  # 记录所有 "Non-Phishing" 节点的总交易数
    non_phishing_total_amount = 0  # 记录所有 "Non-Phishing" 节点的总交易金额

    for (sender_index, sender_label), summary in external_incoming_summary.items():
        count = summary["count"]
        total_amount = summary["total_amount"]

        # 如果是 "Non-Phishing" 节点，则进行汇总，不立即输出
        if sender_label == "Non-Phishing":
            non_phishing_count += 1
            non_phishing_total_transactions += count
            non_phishing_total_amount += total_amount
        else:
            # 立即输出钓鱼节点的信息
            if count > 1:
                avg_amount = total_amount / count
                transaction_details += (f" - From Node {sender_index} ({sender_label}), "
                                        f"Transactions: {count}, "
                                        f"Total Amount: {int(total_amount)}, "
                                        f"Avg Amount: {avg_amount:.1f}\n")
            else:
                transaction_details += (f" - From Node {sender_index} ({sender_label}), "
                                        f"Transactions: {count}, "
                                        f"Amount: {int(total_amount)}\n")

    # 输出汇总的 "Non-Phishing" 节点数据
    if non_phishing_count > 0:
        avg_non_phishing_amount = non_phishing_total_amount / non_phishing_total_transactions if non_phishing_total_transactions > 0 else 0
        transaction_details += (f" - From {non_phishing_count} Non-Phishing Nodes, "
                                f"Total Transactions: {non_phishing_total_transactions}, "
                                f"Total Amount: {int(non_phishing_total_amount)}, "
                                f"Avg Amount: {avg_non_phishing_amount:.1f}\n")

    transaction_details += "**External Outgoing Transactions**\n"
    # 用于汇总所有 "Non-Phishing" 节点的数据
    non_phishing_outgoing_count = 0  # 记录 "Non-Phishing" 节点的数量
    non_phishing_outgoing_total_transactions = 0  # 记录所有 "Non-Phishing" 节点的总交易数
    non_phishing_outgoing_total_amount = 0  # 记录所有 "Non-Phishing" 节点的总交易金额

    for (recipient_index, recipient_label), summary in external_outgoing_summary.items():
        count = summary["count"]
        total_amount = summary["total_amount"]

        # 如果是 "Non-Phishing" 节点，则进行汇总，不立即输出
        if recipient_label == "Non-Phishing":
            non_phishing_outgoing_count += 1
            non_phishing_outgoing_total_transactions += count
            non_phishing_outgoing_total_amount += total_amount
        else:
            # 立即输出钓鱼节点的信息
            if count > 1:
                avg_amount = total_amount / count
                transaction_details += (f" - To Node {recipient_index} ({recipient_label}), "
                                        f"Transactions: {count}, "
                                        f"Total Amount: {int(total_amount)}, "
                                        f"Avg Amount: {avg_amount:.1f}\n")
            else:
                transaction_details += (f" - To Node {recipient_index} ({recipient_label}), "
                                        f"Transactions: {count}, "
                                        f"Amount: {int(total_amount)}\n")

    # 输出汇总的 "Non-Phishing" 节点数据
    if non_phishing_outgoing_count > 0:
        avg_non_phishing_outgoing_amount = non_phishing_outgoing_total_amount / non_phishing_outgoing_total_transactions if non_phishing_outgoing_total_transactions > 0 else 0
        transaction_details += (f" - To {non_phishing_outgoing_count} Non-Phishing Nodes, "
                                f"Total Transactions: {non_phishing_outgoing_total_transactions}, "
                                f"Total Amount: {int(non_phishing_outgoing_total_amount)}, "
                                f"Avg Amount: {avg_non_phishing_outgoing_amount:.1f}\n")

    # 内部交易描述（不展示内部标签）
    internal_transaction_details = "**Internal Transactions**\n"
    sub_index = 1
    for (sender_index, recipient_index, amount), count in internal_transactions_summary.items():
        internal_transaction_details += (f" - From Sub-Node {sub_index} to Sub-Node {sub_index + 1}, "
                                         f"Transactions: {count}, Amount: {int(amount)}\n")
        sub_index += 2

    # 合并描述输出
    node_description = (
        f"-----Node {index_str}-----\n"
        f"*****Node Identity for node {index} ****\n"
        f"PHISHING NODE: {'**YES**, Attention: This node is a potential phishing node and may pose a security risk!' if is_phishing else '**NO**, This node has not been identified as a phishing node and is considered safe.'}\n"
        f"**Node Degree**\n"
        f"External Incoming Transactions: {in_degree}\n"
        f"External Outgoing Transactions: {out_degree}\n"
        f"Total External Transactions: {total_external_transactions}\n"
        f"Total Internal Transactions: {total_internal_transactions}\n"
        f"**Transaction Amounts**\n"
        f"Avg External Incoming Amount: {avg_external_incoming_amount:.1f}\n"
        f"Avg External Outgoing Amount: {avg_external_outgoing_amount:.1f}\n"
        f"Avg Internal Amount: {avg_internal_amount:.1f}\n"
        f"**Degree Centrality**\n"
        f"Degree Centrality: {centrality:.1f}\n"
        f"**Transaction Overview**\n"
        f" - First Transaction Time: {first_transaction_time_str}\n"
        f" - Last Transaction Time: {last_transaction_time_str}\n"
        f"**Average Transaction Frequency**\n"
        f"Avg Transaction Frequency: {avg_transaction_frequency:.1f} transactions/day\n"
        f"{transaction_details}"
        f"{internal_transaction_details}"
        f"*****Node Identity for node {index} ****\n"
        f"PHISHING NODE: {'**YES**, Attention: This node is a potential phishing node and may pose a security risk!' if is_phishing else '**NO**, This node has not been identified as a phishing node and is considered safe.'}\n\n"
    )

    return node_description


def describe_single_node_test(graph, labels, node):
    # 获取节点与索引的映射关系以及节点度中心性
    index_mapping = create_index_mapping(graph)  # 假设该函数返回节点与索引的映射
    degree_centrality = calculate_degree_centrality(graph)  # 假设该函数返回度中心性

    # 节点的基本信息
    is_phishing = labels.get(node, 0)  # 获取合并节点的标签信息
    index = index_mapping.get(node, None)  # 使用 get 方法获取，避免 node 不存在时抛出 KeyError
    index_str = str(index) if index is not None else "N/A"

    # 获取节点的度数信息
    in_degree = graph.in_degree(node) if graph.in_degree(node) is not None else 0
    out_degree = graph.out_degree(node) if graph.out_degree(node) is not None else 0
    total_external_transactions = in_degree + out_degree

    # 获取内部交易边（根据合并节点属性 'internal_edges'）
    internal_edges = graph.nodes[node].get('internal_edges', [])

    # 获取合并节点中的内部节点集合，用于区分内部和外部交易
    internal_nodes = set([node1 for node1, _, _, _ in internal_edges] + [node2 for _, node2, _, _ in internal_edges])

    # 收集所有入边和出边的详细信息
    incoming_edges = [(sender, node, key, data) for sender, _, key, data in graph.in_edges(node, data=True, keys=True)]
    outgoing_edges = [(node, recipient, key, data) for _, recipient, key, data in
                      graph.out_edges(node, data=True, keys=True)]

    # 区分内部交易和外部交易（使用内部节点集合判断）
    internal_edges_combined = [edge for edge in incoming_edges + outgoing_edges
                               if edge[0] == node and edge[1] == node]
    total_internal_transactions = len(internal_edges_combined)

    # 外部交易：发起方或接收方不是内部节点（且不是待测节点自身）
    external_incoming_edges = [edge for edge in incoming_edges if edge[0] not in internal_nodes and edge[0] != node]
    external_outgoing_edges = [edge for edge in outgoing_edges if edge[1] not in internal_nodes and edge[1] != node]

    # 计算内部交易的平均交易金额（所有内部交易合并到一起计算）
    total_internal_amount = sum(edge_data.get('amount', 0) for _, _, _, edge_data in internal_edges_combined)
    avg_internal_amount = total_internal_amount / len(internal_edges_combined) if len(
        internal_edges_combined) > 0 else 0

    # 计算外部交易的平均交易金额
    avg_external_incoming_amount = (
            sum(edge_data.get('amount', 0) for _, _, _, edge_data in external_incoming_edges) / len(
        external_incoming_edges)) if len(external_incoming_edges) > 0 else 0
    avg_external_outgoing_amount = (
            sum(edge_data.get('amount', 0) for _, _, _, edge_data in external_outgoing_edges) / len(
        external_outgoing_edges)) if len(external_outgoing_edges) > 0 else 0

    # 计算交易时间
    transaction_timestamps = [
        edge_data['timestamp'] for _, _, _, edge_data in incoming_edges + outgoing_edges if 'timestamp' in edge_data
    ]

    first_transaction_time = min(transaction_timestamps, default=None)
    last_transaction_time = max(transaction_timestamps, default=None)

    first_transaction_time_str = convert_timestamp(first_transaction_time) if first_transaction_time else 'N/A'
    last_transaction_time_str = convert_timestamp(last_transaction_time) if last_transaction_time else 'N/A'

    # 计算交易频率
    total_transactions = len(incoming_edges) + len(outgoing_edges)  # 总交易数（外部）
    if first_transaction_time and last_transaction_time and first_transaction_time != last_transaction_time:
        total_time_span_seconds = last_transaction_time - first_transaction_time
        total_time_span_days = total_time_span_seconds / 86400  # 将秒转换为天
        avg_transaction_frequency = total_transactions / total_time_span_days if total_time_span_days > 0 else 0
    else:
        avg_transaction_frequency = 0  # 如果没有有效的时间跨度，频率设为 0

    # 节点的度中心性
    centrality = degree_centrality.get(node, 0)  # 如果 centrality 不存在，使用 0 作为默认值

    # 外部交易的汇总信息
    external_incoming_summary = defaultdict(lambda: {"count": 0, "total_amount": 0, "last_timestamp": 0})
    external_outgoing_summary = defaultdict(lambda: {"count": 0, "total_amount": 0, "last_timestamp": 0})

    # 处理外部入边
    for sender, _, key, data in external_incoming_edges:
        sender_index = index_mapping.get(sender, "N/A")
        if sender.startswith('merged_'):
            sender_label = "Unknown"
        else:
            sender_label = "Phishing" if labels.get(sender, 0) == 1 else "Non-Phishing"
        amount = data.get('amount', 0)
        timestamp = data.get('timestamp', 0)
        # 汇总该节点的交易信息
        external_incoming_summary[(sender_index, sender_label)]["count"] += 1
        external_incoming_summary[(sender_index, sender_label)]["total_amount"] += amount
        external_incoming_summary[(sender_index, sender_label)]["last_timestamp"] = max(
            external_incoming_summary[(sender_index, sender_label)]["last_timestamp"], timestamp
        )

    # 处理外部出边
    for _, recipient, key, data in external_outgoing_edges:
        recipient_index = index_mapping.get(recipient, "N/A")
        if recipient.startswith('merged_'):
            recipient_label = "Unknown"
        else:
            recipient_label = "Phishing" if labels.get(recipient, 0) == 1 else "Non-Phishing"
        amount = data.get('amount', 0)
        timestamp = data.get('timestamp', 0)
        # 汇总该节点的交易信息
        external_outgoing_summary[(recipient_index, recipient_label)]["count"] += 1
        external_outgoing_summary[(recipient_index, recipient_label)]["total_amount"] += amount
        external_outgoing_summary[(recipient_index, recipient_label)]["last_timestamp"] = max(
            external_outgoing_summary[(recipient_index, recipient_label)]["last_timestamp"], timestamp
        )

    internal_transactions_summary = defaultdict(int)
    for sender, recipient, key, data in internal_edges_combined:
        sender_index = index_mapping.get(sender, "N/A")
        recipient_index = index_mapping.get(recipient, "N/A")
        amount = data.get('amount', 'N/A')
        transaction_key = (sender_index, recipient_index, amount)
        internal_transactions_summary[transaction_key] += 1

    transaction_details = "**External Incoming Transactions**\n"
    # 用于汇总所有 "Non-Phishing" 节点的数据
    non_phishing_count = 0  # 记录 "Non-Phishing" 节点的数量
    non_phishing_total_transactions = 0  # 记录所有 "Non-Phishing" 节点的总交易数
    non_phishing_total_amount = 0  # 记录所有 "Non-Phishing" 节点的总交易金额

    for (sender_index, sender_label), summary in external_incoming_summary.items():
        count = summary["count"]
        total_amount = summary["total_amount"]

        # 如果是 "Non-Phishing" 节点，则进行汇总，不立即输出
        if sender_label == "Non-Phishing":
            non_phishing_count += 1
            non_phishing_total_transactions += count
            non_phishing_total_amount += total_amount
        else:
            # 立即输出钓鱼节点的信息
            if count > 1:
                avg_amount = total_amount / count
                transaction_details += (f" - From Node {sender_index} ({sender_label}), "
                                        f"Transactions: {count}, "
                                        f"Total Amount: {int(total_amount)}, "
                                        f"Avg Amount: {avg_amount:.1f}\n")
            else:
                transaction_details += (f" - From Node {sender_index} ({sender_label}), "
                                        f"Transactions: {count}, "
                                        f"Amount: {int(total_amount)}\n")

    # 输出汇总的 "Non-Phishing" 节点数据
    if non_phishing_count > 0:
        avg_non_phishing_amount = non_phishing_total_amount / non_phishing_total_transactions if non_phishing_total_transactions > 0 else 0
        transaction_details += (f" - From {non_phishing_count} Non-Phishing Nodes, "
                                f"Total Transactions: {non_phishing_total_transactions}, "
                                f"Total Amount: {int(non_phishing_total_amount)}, "
                                f"Avg Amount: {avg_non_phishing_amount:.1f}\n")

    transaction_details += "**External Outgoing Transactions**\n"
    # 用于汇总所有 "Non-Phishing" 节点的数据
    non_phishing_outgoing_count = 0  # 记录 "Non-Phishing" 节点的数量
    non_phishing_outgoing_total_transactions = 0  # 记录所有 "Non-Phishing" 节点的总交易数
    non_phishing_outgoing_total_amount = 0  # 记录所有 "Non-Phishing" 节点的总交易金额

    for (recipient_index, recipient_label), summary in external_outgoing_summary.items():
        count = summary["count"]
        total_amount = summary["total_amount"]

        # 如果是 "Non-Phishing" 节点，则进行汇总，不立即输出
        if recipient_label == "Non-Phishing":
            non_phishing_outgoing_count += 1
            non_phishing_outgoing_total_transactions += count
            non_phishing_outgoing_total_amount += total_amount
        else:
            # 立即输出钓鱼节点的信息
            if count > 1:
                avg_amount = total_amount / count
                transaction_details += (f" - To Node {recipient_index} ({recipient_label}), "
                                        f"Transactions: {count}, "
                                        f"Total Amount: {int(total_amount)}, "
                                        f"Avg Amount: {avg_amount:.1f}\n")
            else:
                transaction_details += (f" - To Node {recipient_index} ({recipient_label}), "
                                        f"Transactions: {count}, "
                                        f"Amount: {int(total_amount)}\n")

    # 输出汇总的 "Non-Phishing" 节点数据
    if non_phishing_outgoing_count > 0:
        avg_non_phishing_outgoing_amount = non_phishing_outgoing_total_amount / non_phishing_outgoing_total_transactions if non_phishing_outgoing_total_transactions > 0 else 0
        transaction_details += (f" - To {non_phishing_outgoing_count} Non-Phishing Nodes, "
                                f"Total Transactions: {non_phishing_outgoing_total_transactions}, "
                                f"Total Amount: {int(non_phishing_outgoing_total_amount)}, "
                                f"Avg Amount: {avg_non_phishing_outgoing_amount:.1f}\n")


    # # 外部交易描述
    # # 如果有内部交易，则获取第一条交易的时间戳，作为该节点的参考时间
    # reference_timestamp = internal_edges[0][3]['timestamp'] if internal_edges else None
    # six_months_in_seconds = recent_month * 30 * 24 * 60 * 60

    # transaction_details = "**External Incoming Transactions**\n"
    # for (sender_index, sender_label), summary in external_incoming_summary.items():
    #     count = summary["count"]
    #     total_amount = summary["total_amount"]
    #     last_timestamp = summary["last_timestamp"]
    #     if reference_timestamp and (reference_timestamp - last_timestamp) > six_months_in_seconds:
    #         continue
    #     if count > 1:
    #         avg_amount = total_amount / count
    #         transaction_details += (f" - From Node {sender_index} ({sender_label}), "
    #                                 f"Transactions: {count}, "
    #                                 f"Total Amount: {int(total_amount)}, "
    #                                 f"Avg Amount: {avg_amount:.1f}\n")
    #     else:
    #         transaction_details += (f" - From Node {sender_index} ({sender_label}), "
    #                                 f"Transactions: {count}, "
    #                                 f"Amount: {int(total_amount)}\n")
    #
    # transaction_details += "**External Outgoing Transactions**\n"
    # for (recipient_index, recipient_label), summary in external_outgoing_summary.items():
    #     count = summary["count"]
    #     total_amount = summary["total_amount"]
    #     last_timestamp = summary["last_timestamp"]
    #     if reference_timestamp and (reference_timestamp - last_timestamp) > six_months_in_seconds:
    #         continue
    #     if count > 1:
    #         avg_amount = total_amount / count
    #         transaction_details += (f" - To Node {recipient_index} ({recipient_label}), "
    #                                 f"Transactions: {count}, "
    #                                 f"Total Amount: {int(total_amount)}, "
    #                                 f"Avg Amount: {avg_amount:.1f}\n")
    #     else:
    #         transaction_details += (f" - To Node {recipient_index} ({recipient_label}), "
    #                                 f"Transactions: {count}, "
    #                                 f"Amount: {int(total_amount)}\n")

    # 内部交易描述（不展示内部标签）
    internal_transaction_details = "**Internal Transactions**\n"
    sub_index = 1
    for (sender_index, recipient_index, amount), count in internal_transactions_summary.items():
        internal_transaction_details += (f" - From Sub-Node {sub_index} to Sub-Node {sub_index + 1}, "
                                         f"Transactions: {count}, Amount: {int(amount)}\n")
        sub_index += 2

    # 合并描述输出
    node_description = (
        f"-----Node {index_str}-----\n"
        f"**Node Degree**\n"
        f"External Incoming Transactions: {in_degree}\n"
        f"External Outgoing Transactions: {out_degree}\n"
        f"Total External Transactions: {total_external_transactions}\n"
        f"Total Internal Transactions: {total_internal_transactions}\n"
        f"**Transaction Amounts**\n"
        f"Avg External Incoming Amount: {avg_external_incoming_amount:.1f}\n"
        f"Avg External Outgoing Amount: {avg_external_outgoing_amount:.1f}\n"
        f"Avg Internal Amount: {avg_internal_amount:.1f}\n"
        f"**Degree Centrality**\n"
        f"Degree Centrality: {centrality:.1f}\n"
        f"**Transaction Overview**\n"
        f" - First Transaction Time: {first_transaction_time_str}\n"
        f" - Last Transaction Time: {last_transaction_time_str}\n"
        f"**Average Transaction Frequency**\n"
        f"Avg Transaction Frequency: {avg_transaction_frequency:.1f} transactions/day\n"
        f"{transaction_details}"
        f"{internal_transaction_details}"
    )

    return node_description, is_phishing


def convert_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def create_index_mapping(graph):
    return {node: index for index, node in enumerate(graph.nodes)}


def calculate_degree_centrality(graph):
    total_nodes = len(graph.nodes)
    return {node: (degree / (total_nodes - 1)) for node, degree in dict(graph.degree()).items()}


if __name__ == '__main__':
    dataset_dir = f'../../dataset/clustered_graphs/ZipZap/200/delay_5'
    graph_test_file_path = f'{dataset_dir}/8/test.pkl'
    labels_test_file_path = f'{dataset_dir}/8/test_labels.json'
    graph_test = load_graph(graph_test_file_path)
    labels_test = load_json(labels_test_file_path)
    for i, node in enumerate(graph_test.nodes):
        test_node_detail, is_phishing = describe_single_node_test(graph_test, labels_test, node, 6)
        print(test_node_detail)
        print(f"is_phishing : {is_phishing}")
        break

    # graph_test_file_path = f'{dataset_dir}/1/train.pkl'
    # labels_test_file_path = f'{dataset_dir}/1/train_labels.json'
    # graph_test = load_graph(graph_test_file_path)
    # labels_test = load_labels(labels_test_file_path)
    #
    # for i, node in enumerate(graph_test.nodes):
    #     train_node_detail = describe_single_node_train(graph_test, labels_test, node)
    #     print(train_node_detail)
    #     break
