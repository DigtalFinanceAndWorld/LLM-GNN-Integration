import json
import pickle
from datetime import datetime
from collections import defaultdict
import numpy as np


def load_graph(file_path):
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def load_json(file_path):
    with open(file_path, 'r') as f:
        labels = json.load(f)
    return labels


def describe_single_node_test(graph, labels, node):
    # 获取节点与索引的映射关系以及节点度中心性
    index_mapping = create_index_mapping(graph)  # 假设该函数返回节点与索引的映射
    centrality = calculate_node_degree_centrality(graph, node)  # 假设该函数返回度中心性

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
                      
    # 初始化余额变化记录
    balance_changes = []

    # 计算初始余额和每笔交易的累积余额变化
    current_balance = 0  # 初始余额设为 0
    for edge in sorted(incoming_edges + outgoing_edges, key=lambda x: x[3].get('timestamp', 0)):
        amount = edge[3].get('amount', 0)
        if edge in incoming_edges:
            current_balance += amount  # 收到资金增加余额
        else:
            current_balance -= amount  # 转出资金减少余额
        balance_changes.append(current_balance)

    # 计算最高和最低余额
    max_balance = max(balance_changes, default=0)
    min_balance = min(balance_changes, default=0)
    balance_fluctuation = max_balance - min_balance
    balance_std_dev = np.std(balance_changes)
    balance_drops = len([1 for i in range(1, len(balance_changes)) if balance_changes[i] < balance_changes[i-1] * 0.5])  # 余额减少超过50%         

    # 用于存储大额交易的每日聚合信息
    daily_large_transactions = defaultdict(list)
    daily_large_transactions_post = defaultdict(list)
    processed_timestamps = set()  # 用于记录已处理过的时间戳

    # 设置大额交易阈值（可根据需要调整）
    large_transaction_threshold = 10

    # 遍历所有入边，分析在大额交易后 10 小时内的活动
    for edge in sorted(incoming_edges, key=lambda x: x[3].get('timestamp', 0)):
        amount = edge[3].get('amount', 0)
        timestamp = edge[3].get('timestamp', 0)
    
        # 检查是否为大额交易且未被处理
        if amount >= large_transaction_threshold and timestamp not in processed_timestamps:
            processed_timestamps.add(timestamp)  # 将时间戳添加到已处理集合中
        
            # 查找该大额交易后的 10 小时内出边活动
            for out_edge in outgoing_edges:
                out_timestamp = out_edge[3].get('timestamp', 0)
                if 0 <= out_timestamp - timestamp <= 36000:  # 10小时内
                    # 记录大额交易的日期和金额
                    date = convert_timestamp(timestamp).split(" ")[0]  # 只提取日期
                    daily_large_transactions[date].append(amount)
                    daily_large_transactions_post[date].append(1)
                    break  # 只记录一次活动，避免重复
            
    # 计算大额交易的所有金额，以便计算 min, max, avg
    all_large_transactions = [amount for amounts in daily_large_transactions.values() for amount in amounts]

    # 计算大额交易的总交易次数
    total_large_transactions = len(all_large_transactions)

    # 计算大额交易中的最小金额
    min_large_amount = min(all_large_transactions) if all_large_transactions else 0

    # 计算大额交易中的最大金额
    max_large_amount = max(all_large_transactions) if all_large_transactions else 0

    # 计算大额交易的平均金额
    avg_large_amount = sum(all_large_transactions) / total_large_transactions if total_large_transactions > 0 else 0
        
    # 计算每日大额交易的总次数并求平均
    total_large_transactions_per_day = {
        date: sum(amounts) for date, amounts in daily_large_transactions.items()
    }
    avg_large_transactions_per_day = sum(total_large_transactions_per_day.values()) / len(total_large_transactions_per_day) if total_large_transactions_per_day else 0

    # 计算大额交易后发生的交易次数并求平均
    avg_transactions_per_day_after_large_txn = {
        date: sum(activities) for date, activities in daily_large_transactions_post.items()
    }
    avg_transactions_per_day_after_large_txn_value = sum(avg_transactions_per_day_after_large_txn.values()) / len(avg_transactions_per_day_after_large_txn) if avg_transactions_per_day_after_large_txn else 0


    # 构建按日期的活动描述
    activity_description = "**Specific Time Point Activity Summary**\n"
    for date, amounts in daily_large_transactions.items():
        min_amount = min(amounts)
        max_amount = max(amounts)
        transaction_count = len(amounts)
        activity_description += (
            f" - Date: {date}, Large Transactions: {transaction_count}, "
            f"Amount Range: {min_amount:.2f} - {max_amount:.2f}\n"
        )

    # 计算整体统计摘要
    all_amounts = [amount for amounts in daily_large_transactions.values() for amount in amounts]
    total_large_transactions = len(all_amounts)
    min_large_amount = min(all_amounts) if all_amounts else 0
    max_large_amount = max(all_amounts) if all_amounts else 0
    avg_large_amount = sum(all_amounts) / total_large_transactions if total_large_transactions > 0 else 0

    # 添加总体统计摘要
    activity_description += "**Overall Large Transaction Summary**\n"
    activity_description += (
        f"Total Large Transactions: {total_large_transactions}\n"
        f"Amount Range: {min_large_amount:.2f} - {max_large_amount:.2f}\n"
        f"Average Large Transaction Amount: {avg_large_amount:.2f}\n"
    )

    # 如果没有大额交易活动
    if not daily_large_transactions:
        activity_description = "No significant activity following large incoming transactions\n"   

    # 设置小额交易的阈值
    small_transaction_threshold = 1.0

    # 初始化统计计数
    small_incoming_count = 0
    small_outgoing_count = 0
    zero_incoming_count = 0
    zero_outgoing_count = 0

    # 遍历所有入边和出边，计算小额和零值交易
    for edge in incoming_edges:
        amount = edge[3].get('amount', 0)
        if amount == 0:
            zero_incoming_count += 1
        elif amount < small_transaction_threshold:
            small_incoming_count += 1

    for edge in outgoing_edges:
        amount = edge[3].get('amount', 0)
        if amount == 0:
            zero_outgoing_count += 1
        elif amount < small_transaction_threshold:
            small_outgoing_count += 1

    # 计算比例
    total_incoming = len(incoming_edges)
    total_outgoing = len(outgoing_edges)
    inflow_outflow_ratio = total_incoming / total_outgoing if total_outgoing != 0 else 0

    zero_incoming_ratio = zero_incoming_count / total_incoming if total_incoming > 0 else 0
    small_incoming_ratio = small_incoming_count / total_incoming if total_incoming > 0 else 0
    zero_outgoing_ratio = zero_outgoing_count / total_outgoing if total_outgoing > 0 else 0
    small_outgoing_ratio = small_outgoing_count / total_outgoing if total_outgoing > 0 else 0
    
    # 添加交易类别统计描述
    transaction_category_description = f"**Transaction Category Statistics**\n" \
                                       f"Zero Incoming Transactions: {zero_incoming_count} ({zero_incoming_ratio:.2%})\n" \
                                       f"Small Incoming Transactions (<{small_transaction_threshold}): " \
                                       f"{small_incoming_count} ({small_incoming_ratio:.2%})\n" \
                                       f"Zero Outgoing Transactions: {zero_outgoing_count} ({zero_outgoing_ratio:.2%})\n" \
                                       f"Small Outgoing Transactions (<{small_transaction_threshold}): " \
                                       f"{small_outgoing_count} ({small_outgoing_ratio:.2%})\n"


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

    high_count = 6
    high_total_amount = 10
    transaction_details = "**External Incoming Transactions**\n"
    phishing_incoming_count = 0  # 记录 "Phishing" 节点的数量
    phishing_incoming_total_transactions = 0  # 记录所有 "Phishing" 节点的总交易数
    
    non_phishing_incoming_count = 0  # 记录 "Non-Phishing" 节点的数量
    non_phishing_incoming_total_transactions = 0  # 记录所有 "Phishing" 节点的总交易数
    non_phishing_incoming_total_amount = 0  # 记录 "Non-Phishing" 节点的总交易
    
    unknown_incoming_count = 0  # 记录所有 "unknown" 节点的数量
    unknown_incoming_total_transactions = 0  # 记录所有 "unknown" 节点的总交易数
    
    for (sender_index, sender_label), summary in external_incoming_summary.items():
        count = summary["count"]
        total_amount = summary["total_amount"]

        # 如果是 "Non-Phishing" 节点，则进行汇总，不立即输出
        if sender_label == "Phishing":
            phishing_incoming_count += 1
            phishing_incoming_total_transactions += count
        elif sender_label == "Non-Phishing":
            non_phishing_incoming_count += 1
            non_phishing_incoming_total_transactions += count
            non_phishing_incoming_total_amount += total_amount
        elif sender_label == "Unknown":
            unknown_incoming_count += 1
            unknown_incoming_total_transactions += count


    transaction_details += "**External Outgoing Transactions**\n"
    phishing_outgoing_count = 0  # 记录 "Non-Phishing" 节点的数量
    phishing_outgoing_total_transactions = 0  # 记录所有 "Non-Phishing" 节点的总交易数
    
    non_phishing_outgoing_count = 0  # 记录 "Non-Phishing" 节点的数量
    non_phishing_outgoing_total_transactions = 0  # 记录所有 "Non-Phishing" 节点的总交易数
    non_phishing_outgoing_total_amount = 0  # 记录所有 "Non-Phishing" 节点的总交易金额
    
    unknown_outgoing_count = 0  # 记录 "unknown" 节点的数量
    unknown_outgoing_total_transactions = 0  # 记录所有 "unknown" 节点的总交易数
    
    for (recipient_index, recipient_label), summary in external_outgoing_summary.items():
        count = summary["count"]
        total_amount = summary["total_amount"]

        # 如果是 "Non-Phishing" 节点，则进行汇总，不立即输出
        if recipient_label == "Phishing":
            phishing_outgoing_count += 1
            phishing_outgoing_total_transactions += count
        elif recipient_label == "Non-Phishing":
            non_phishing_outgoing_count += 1
            non_phishing_outgoing_total_transactions += count
            non_phishing_outgoing_total_amount += total_amount
        elif recipient_label == "Unknown":
            unknown_outgoing_count += 1
            unknown_outgoing_total_transactions += count

    
    # 进行浮动数值格式化并存储到变量中
    formatted_avg_external_incoming_amount = round(avg_external_incoming_amount, 1)
    formatted_avg_external_outgoing_amount = round(avg_external_outgoing_amount, 1)
    formatted_centrality = round(centrality, 5)
    formatted_avg_internal_amount = round(avg_internal_amount, 1)
    formatted_avg_transaction_frequency = round(avg_transaction_frequency, 1)
    formatted_max_balance = round(max_balance, 2)
    formatted_min_balance = round(min_balance, 2)
    formatted_balance_fluctuation = round(balance_fluctuation, 2)
    formatted_balance_std_dev = round(balance_std_dev, 2)
    formatted_inflow_outflow_ratio = round(inflow_outflow_ratio, 2)
    formatted_zero_incoming_ratio = round(zero_incoming_ratio, 2)
    formatted_small_incoming_ratio = round(small_incoming_ratio, 2)
    formatted_zero_outgoing_ratio = round(zero_outgoing_ratio, 2)
    formatted_small_outgoing_ratio = round(small_outgoing_ratio, 2)
    formatted_total_large_transactions = round(total_large_transactions, 2)
    formatted_min_large_amount = round(min_large_amount, 2)
    formatted_max_large_amount = round(max_large_amount, 2)
    formatted_avg_large_amount = round(avg_large_amount, 2)
    formatted_avg_large_transactions_per_day = round(avg_large_transactions_per_day, 2)
    formatted_avg_transactions_per_day_after_large_txn_value = round(avg_transactions_per_day_after_large_txn_value, 2)


    # 现在可以将格式化后的值添加到字典中
    info_dict = {
	"in_degree": in_degree,
	"out_degree": out_degree,
	"total_external_transactions": total_external_transactions,	
	"avg_external_incoming_amount": formatted_avg_external_incoming_amount,
	"avg_external_outgoing_amount": formatted_avg_external_outgoing_amount,
	"centrality": formatted_centrality,
	"avg_transaction_frequency": formatted_avg_transaction_frequency,
	
	"total_internal_transactions": total_internal_transactions,
	"avg_internal_amount": formatted_avg_internal_amount,
	
	"max_balance": formatted_max_balance,
	"min_balance": formatted_min_balance,
	"balance_fluctuation": formatted_balance_fluctuation,
	"balance_std_dev": formatted_balance_std_dev,
	"inflow_outflow_ratio": formatted_inflow_outflow_ratio,
	"balance_drops": balance_drops,
	
	"zero_incoming_count": zero_incoming_count,
	"zero_incoming_ratio": formatted_zero_incoming_ratio,
	"small_incoming_count": small_incoming_count,
	"small_incoming_ratio": formatted_small_incoming_ratio,
	"zero_outgoing_count": zero_outgoing_count,
	"zero_outgoing_ratio": formatted_zero_outgoing_ratio,
	"small_outgoing_count": small_outgoing_count,
	"small_outgoing_ratio": formatted_small_outgoing_ratio,
	
	"total_large_transactions": formatted_total_large_transactions,
	"min_large_amount": formatted_min_large_amount,
	"max_large_amount": formatted_max_large_amount,
	"avg_large_amount": formatted_avg_large_amount,
	"avg_large_transactions_per_day": formatted_avg_large_transactions_per_day,
	"avg_transactions_per_day_after_large_txn_value": formatted_avg_transactions_per_day_after_large_txn_value,
	
	"phishing_incoming_count": phishing_incoming_count,
	"phishing_incoming_total_transactions": phishing_incoming_total_transactions,
	"phishing_outgoing_count": phishing_outgoing_count,
	"phishing_outgoing_total_transactions": phishing_outgoing_total_transactions,
	"non_phishing_incoming_count": non_phishing_incoming_count,
	"non_phishing_incoming_total_transactions": non_phishing_incoming_total_transactions,
	"non_phishing_outgoing_count": non_phishing_outgoing_count,
	"non_phishing_outgoing_total_transactions": non_phishing_outgoing_total_transactions,
	"unknown_incoming_count": unknown_incoming_count,
	"unknown_incoming_total_transactions": unknown_incoming_total_transactions,
	"unknown_outgoing_count": unknown_outgoing_count,
	"unknown_outgoing_total_transactions": unknown_outgoing_total_transactions,
	
	"is_phishing": is_phishing
    }
    
    # 合并描述输出
    node_description = f"""
{{
  "node_information": {{
    "node_index": "{index_str}",
    "sections": [
      {{
        "title": "1. Node Transaction Activity",
        "description": "This node has conducted transactions both incoming and outgoing, reflecting its activity level in the network. Its total number of external transactions shows its engagement, while its centrality score indicates how influential it is in the overall network structure.",
        "inference": "A node with high transaction activity and centrality may play a significant role in the network. However, such behavior could also signal potential risks if unusual patterns are observed.",
        "details": [
          "The node has received **{info_dict.get('in_degree', 'N/A')}** transactions and sent out **{info_dict.get('out_degree', 'N/A')}** transactions.",
          "It has completed **{info_dict.get('total_external_transactions', 'N/A')}** external transactions in total.",
          "The average amount of incoming external transactions is **{info_dict.get('avg_external_incoming_amount', 'N/A')}**, while outgoing transactions average **{info_dict.get('avg_external_outgoing_amount', 'N/A')}**.",
          "On average, the node processes **{info_dict.get('avg_transaction_frequency', 'N/A')}** transactions per day.",
          "The centrality score of **{info_dict.get('centrality', 'N/A')}** indicates its influence within the network."
        ],
        "objective": "Determine if the node demonstrates abnormal levels of activity or network influence, which could indicate potential phishing behavior."
      }},
      {{
        "title": "2. Transaction Amount and Distribution",
        "description": "The node's transactions include a mix of zero-value and small-value amounts. These could indicate normal activity, but an unusually high proportion of such transactions might point to attempts at obfuscation or simulation of legitimate behavior.",
        "inference": "High volumes of zero-value or small-value transactions can be indicative of efforts to confuse tracking systems or mask real intentions.",
        "details": [
          "The node has conducted **{info_dict.get('zero_incoming_count', 'N/A')}** incoming zero-value transactions, making up **{info_dict.get('zero_incoming_ratio', 'N/A')}%** of all incoming transactions.",
          "It has received **{info_dict.get('small_incoming_count', 'N/A')}** small-value (<1) incoming transactions, which represent **{info_dict.get('small_incoming_ratio', 'N/A')}%** of the total.",
          "On the outgoing side, **{info_dict.get('zero_outgoing_count', 'N/A')}** zero-value transactions account for **{info_dict.get('zero_outgoing_ratio', 'N/A')}%**, and **{info_dict.get('small_outgoing_count', 'N/A')}** small-value transactions make up **{info_dict.get('small_outgoing_ratio', 'N/A')}%** of outgoing activity."
        ],
        "objective": "Assess whether the node uses small or zero-value transactions to simulate normal activity or obscure its actual behavior."
      }},
      {{
        "title": "3. Balance Fluctuation",
        "description": "The node’s balance changes over time reveal how it manages funds. Large and frequent fluctuations, especially sharp drops, may indicate rapid fund movement or attempts to clear balances quickly.",
        "inference": "Nodes with significant balance fluctuations might be involved in transferring or distributing funds rapidly to evade detection.",
        "details": [
          "The highest balance recorded for this node was **{info_dict.get('max_balance', 'N/A')}**, while the lowest was **{info_dict.get('min_balance', 'N/A')}**.",
          "Over time, the balance fluctuated by **{info_dict.get('balance_fluctuation', 'N/A')}**, with **{info_dict.get('balance_drops', 'N/A')}** instances of drops exceeding 50%.",
          "The balance's standard deviation is **{info_dict.get('balance_std_dev', 'N/A')}**, reflecting its variability.",
          "The ratio of incoming to outgoing funds is **{info_dict.get('inflow_outflow_ratio', 'N/A')}**."
        ],
        "objective": "Identify whether significant balance changes suggest attempts to hide or redistribute funds."
      }},
      {{
        "title": "4. Transaction Frequency and Timing",
        "description": "This section evaluates how the node handles transactions after large inflows of funds. Patterns of rapid outgoing transactions may suggest attempts to obscure fund origins or destinations.",
        "inference": "Bursts of transactions following large inflows often indicate efforts to redistribute funds quickly and avoid detection.",
        "details": [
          "The node performed **{info_dict.get('total_large_transactions', 'N/A')}** large transactions in total.",
          "The smallest large transaction amount was **{info_dict.get('min_large_amount', 'N/A')}**, while the largest was **{info_dict.get('max_large_amount', 'N/A')}**.",
          "On average, large transactions were valued at **{info_dict.get('avg_large_amount', 'N/A')}**, and the node conducted **{info_dict.get('avg_large_transactions_per_day', 'N/A')}** large transactions daily.",
          "Following large inflows, the node averaged **{info_dict.get('avg_transactions_per_day_after_large_txn_value', 'N/A')}** transactions per day."
        ],
        "objective": "Detect whether rapid outgoing activity following large inflows could indicate fund redistribution or obfuscation."
      }},
      {{
        "title": "5. Counterparty Analysis",
        "description": "This node interacts with various counterparties, including known phishing nodes, unknown nodes, and non-phishing nodes. Frequent interactions with risky entities may indicate malicious intent.",
        "inference": "High transaction volumes with phishing or unknown nodes may suggest involvement in fraudulent networks.",
        "details": [
          "It has received funds from **{info_dict.get('phishing_incoming_count', 'N/A')}** phishing nodes across **{info_dict.get('phishing_incoming_total_transactions', 'N/A')}** transactions.",
          "The node sent funds to **{info_dict.get('phishing_outgoing_count', 'N/A')}** phishing nodes in **{info_dict.get('phishing_outgoing_total_transactions', 'N/A')}** transactions.",
          "Funds were received from **{info_dict.get('unknown_incoming_count', 'N/A')}** unknown nodes in **{info_dict.get('unknown_incoming_total_transactions', 'N/A')}** transactions.",
          "Outgoing transactions to unknown nodes totaled **{info_dict.get('unknown_outgoing_count', 'N/A')}** interactions across **{info_dict.get('unknown_outgoing_total_transactions', 'N/A')}** transactions."
        ],
        "objective": "Evaluate whether the node’s interactions suggest its involvement in a phishing network."
      }},
      {{
        "title": "6. Internal Transactions",
        "description": "Internal transactions involve moving funds between addresses controlled by the same entity. These transfers might aim to obscure fund origins or simulate normal activity.",
        "inference": "High-frequency or low-value internal transactions often point to efforts to self-circulate funds and obscure their flow.",
        "details": [
          "The node carried out **{info_dict.get('total_internal_transactions', 'N/A')}** internal transactions.",
          "The average value of these internal transactions was **{info_dict.get('avg_internal_amount', 'N/A')}**."
        ],
        "objective": "Determine whether internal transactions indicate attempts to conceal fund flows or simulate legitimate activity."
      }}
    ]
  }}
}}
"""



    return node_description, is_phishing, info_dict


def convert_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def create_index_mapping(graph):
    return {node: index for index, node in enumerate(graph.nodes)}


def calculate_node_degree_centrality(graph, node):
    """计算某个节点的度中心性（入度 + 出度）。"""
    if len(graph) <= 1:
        return 1  # 如果图中只有一个节点，它的中心性为1

    s = 1.0 / (len(graph) - 1.0)  # 标准化系数

    # 获取该节点的入度和出度
    in_deg = graph.in_degree(node)
    out_deg = graph.out_degree(node)

    # 计算该节点的度中心性
    centrality = (in_deg + out_deg) * s
    return centrality
