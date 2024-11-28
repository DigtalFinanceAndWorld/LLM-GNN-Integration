from describe_single_node_testing import describe_single_node_train


def purest_rank(nodes, current_labels, nodemap_train):
    merged_node_to_original_nodes = {}
    merged_num_to_node_name = {}
    for node in nodes:
        if node.startswith('merged_'):
            merged_node_num = int(node[len('merged_'):])
            merged_num_to_node_name[merged_node_num] = node
            merged_node_to_original_nodes[node] = set()
        else:
            print(f"[WARNING] 节点 {node} 不是合并节点，可能需要检查")

    for address, merged_num in nodemap_train.items():
        merged_node_name = merged_num_to_node_name.get(merged_num)
        if merged_node_name:
            merged_node_to_original_nodes[merged_node_name].add(address)

    p_list = {}
    for node, address_set in merged_node_to_original_nodes.items():
        p = 0
        for address in address_set:
            p += current_labels[address]
        p = p / len(address_set)
        p_list[node] = p
    p_sort = sorted(p_list.items(), key=lambda x: x[1], reverse=True)
    return p_sort


def generate_purest_detail(add_histroy_graph, add_histroy_labels, graph_train, current_labels, nodemap_train):
    # 筛选符合条件的节点
    phishing_nodes = [node for node in graph_train.nodes if add_histroy_labels.get(node, 0) == 1]
    non_phishing_nodes = [node for node in graph_train.nodes if add_histroy_labels.get(node, 0) == 0]

    phishing_p_sort = purest_rank(phishing_nodes, current_labels, nodemap_train)
    print(f"phishing_p_sort: {phishing_p_sort}")

    # 生成自然语言描述
    node_descriptions = {}
    phishing_samples = []
    non_phishing_samples = []
    min_description_length = 5000
    max_description_length = 20000

    # top3
    n = 3
    for node, p in phishing_p_sort[:n]:
        description = describe_single_node_train(add_histroy_graph, add_histroy_labels, node)
        print(f"add train example: phishing_nodes {node}, p = {p} ...")
        phishing_samples.append(node)
        node_descriptions[node] = description

    for node in non_phishing_nodes:
        description = describe_single_node_train(add_histroy_graph, add_histroy_labels, node)
        if min_description_length <= len(description) <= max_description_length:
            print(f"add train example: non_phishing_nodes {node} ...")
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
