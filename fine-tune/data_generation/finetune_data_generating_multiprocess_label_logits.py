from multiprocessing import Process, Manager
import re
import orjson
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from prompt.single_expert.add_history import add_history
from prompt.single_expert.describe_single_node_testing_token_4096 import *


def process_node(node, graph, labels):
    # 获取节点描述及其标签
    node_description, _ = describe_single_node_test(graph, labels, node)
    label = labels[node]

    # 返回结构化 JSON 格式，确保模型理解明确任务
    return {
        "instruction": """
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
""",
        "input": f"""
### Target Node Information
{node_description}
""",
        "output": f"""
### Output Label
{label}
"""
    }


def writer_process(queue, result_file_path, writer_id):
    """专门负责写入单个文件的进程，从队列中逐条获取数据写入到文件中"""
    print(f"Writer Process {writer_id} started, saving data to {result_file_path}...")

    with open(result_file_path, 'w') as f:
        f.write('[\n')  # 写入 JSON 数组的开头
        is_first = True  # 用于处理逗号
        count = 0

        while True:
            try:
                item = queue.get()
                if item is None:  # 检查退出信号
                    break
                if not is_first:
                    f.write(',\n')  # 如果不是第一个元素，则写入逗号换行
                f.write(orjson.dumps(item).decode('utf-8'))
                is_first = False
                count += 1
                print(f"Writer {writer_id}: Wrote {count} items.")
            except EOFError:
                break

        f.write('\n]')  # 写入 JSON 数组的结尾
    print(f"Writer Process {writer_id} completed writing to {result_file_path}.")


def process_cluster(index, dataset_name, delay, dataset_dir, queue):
    """处理每个 cluster，并将结果放入指定的队列中"""
    start_date_file_path = f'../../dataset/{dataset_name}/start_times.json'
    train_start_date = load_json(start_date_file_path)["train"][index]
    graph_all_file_path = f'../../dataset/{dataset_name}/data/{dataset_name}.pkl'
    graph_all = load_graph(graph_all_file_path)

    graph_train_file_path = f'{dataset_dir}/{index}/train.pkl'
    labels_train_file_path = f'{dataset_dir}/{index}/train_labels.json'
    nodemap_train_file_path = f'{dataset_dir}/{index}/train_nodemap.json'
    graph_train = load_graph(graph_train_file_path)
    labels_train = load_json(labels_train_file_path)
    nodemap_train = load_json(nodemap_train_file_path)

    add_histroy_train_graph, add_histroy_train_labels = add_history(graph_all, graph_train, labels_train, nodemap_train,
                                                                    train_start_date, delay)

    # 逐个处理 sampled_nodes 中的节点，并将结果放入对应的 queue 中
    print(f"graph_train nodes: {len(graph_train)}")
    for node in graph_train:
        result = process_node(node, add_histroy_train_graph, add_histroy_train_labels)
        # print(result)
        queue.put(result)  # 将处理结果放入队列中
    print(f"Cluster {index} processing completed, all nodes have been added to queue.")


def main():
    num_workers = 8  # 控制同时进行的任务数量
    dataset_name = "MulDiGraph"
    multiple_list = ["500", "1000", "100"]
    delay = 5
    result_dir = f'../../dataset/finetune/finetune_label_logits_4096/'
    os.makedirs(result_dir, exist_ok=True)
    manager = Manager()  # 创建Manager对象，用于管理队列

    for multiple in multiple_list:
        dataset_dir = f'../../dataset/clustered_graphs/{dataset_name}/{multiple}/delay_{delay}'
        if os.path.exists(dataset_dir):
            print(f"Dir exists: {dataset_dir}")
            dataset_list = os.listdir(dataset_dir)
        else:
            print(f"Dir does not exists: {dataset_dir}")
            continue

        def natural_sort_key(filename):
            return [int(s) if s.isdigit() else s for s in re.split('([0-9]+)', filename)]

        # 确保 sorted_dataset_list 中的元素是按自然顺序排列
        sorted_dataset_list = sorted(dataset_list, key=natural_sort_key)
        print(f"Dataset_list: {sorted_dataset_list}")

        # 创建每个 `sorted_dataset_list` 对应的队列和写入进程
        queue_list = [manager.Queue() for _ in range(len(sorted_dataset_list))]
        writer_processes = []

        # 创建写入进程，每个 `sorted_dataset_list` 项对应一个写入进程，并严格与其 index 对应
        for idx, index in enumerate(sorted_dataset_list):
            result_file_path = f'{result_dir}/{dataset_name}_delay_{delay}_{multiple}_{index}.json'
            writer = Process(target=writer_process, args=(queue_list[idx], result_file_path, index))
            writer.start()
            writer_processes.append(writer)

        # 使用 ProcessPoolExecutor 控制并发数量
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_cluster, index, dataset_name, delay, dataset_dir, queue_list[i])
                for i, index in enumerate(sorted_dataset_list)
            ]

            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()  # 如果有异常，会在这里抛出
                except Exception as e:
                    print(f"Task raised an exception: {e}")

        # 关闭所有队列，发送退出信号，等待所有写入进程完成
        for queue in queue_list:
            queue.put(None)  # 发送退出信号
        for writer in writer_processes:
            writer.join()  # 等待所有写入进程完成

        print(f"Fine-tuning datasets saved to files in {result_dir}.")


if __name__ == '__main__':
    main()
