from multiprocessing import Process, Manager
import re
import orjson
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from prompt.single_expert.add_history import add_history
from prompt.single_expert.describe_single_node_testing_token_cot_json import *
from select_analysis import get_cot_3_analysis


def get_json(label, analysis):
    # 使用正则表达式提取 ```json 和 ``` 之间的内容
    json_match = re.search(r"```json\s*(.*?)\s*```", analysis.strip(), re.DOTALL)

    if json_match:
        json_string = json_match.group(1)  # 提取 JSON 内容
        try:
            # 将 JSON 字符串解析为 Python 字典
            data = json.loads(json_string)
            # 添加 "label": {label} 到 JSON 数据中
            data = {
                "label": label,
                **data
            }
            return json.dumps(data, indent=2)  # 格式化输出
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
    else:
        print("未找到 JSON 内容")
                                            


def process_node(node, graph, labels, analysis):
    # 获取节点描述及其标签
    node_description, _, _ = describe_single_node_test(graph, labels, node)
    label = labels[node]
    output = get_json(label, analysis)
    print("process node !!!")
    print(output)

    # 返回结构化 JSON 格式，确保模型理解明确任务
    return {
        "instruction": """
{
  "instruction": {
    "description": "You are a blockchain analyst specializing in identifying fraudulent behavior within the Ethereum network. Your task is to analyze the transaction data of a **Target Node** and classify it as either a **Phishing Node** or **Non-Phishing Node**.",
    "classification": {
      "labels": {
        "1": "Phishing Node",
        "0": "Non-Phishing Node"
      },
      "requirement": "This is a **binary classification problem**, requiring you to return one of the specified labels."
    }
  },
  "task": "You are a blockchain analyst specializing in identifying fraudulent behavior within the Ethereum network. I will provide you with information about a **Target Node** and its true label (**Phishing** or **Non-phishing**). Your task is to analyze the node's behavior and explain why it has been classified with the given label.",
  "task_details": {
    "methodology": "Use the **Chain-of-Thought (CoT)** methodology to reason through your analysis, combining observations, logical inference, and conclusions.",
    "dimensions": [
      {
        "dimension": "Node Transaction Activity",
        "steps": [
          {
            "step": "Observation",
            "description": "Evaluate the node's in-degree and out-degree, focusing on its overall activity."
          },
          {
            "step": "Inference",
            "description": "A high number of total external transactions suggests significant activity, which, combined with high centrality, could indicate risky behavior."
          },
          {
            "step": "Conclusion",
            "description": "Determine if the node's transaction activity is abnormally high and whether its central role in the network raises concerns."
          }
        ]
      },
      {
        "dimension": "Transaction Amount and Distribution",
        "steps": [
          {
            "step": "Observation",
            "description": "Examine the frequency and ratio of zero-value and small-value transactions for both incoming and outgoing activities."
          },
          {
            "step": "Inference",
            "description": "High proportions of such transactions often signify obfuscation or deceptive intent, a pattern frequently observed in phishing nodes."
          },
          {
            "step": "Conclusion",
            "description": "Determine if these patterns indicate deliberate efforts to confuse tracking mechanisms or simulate legitimate activity."
          }
        ]
      },
      {
        "dimension": "Balance Fluctuation",
        "steps": [
          {
            "step": "Observation",
            "description": "Analyze the node's balance for sharp changes, including significant inflows and rapid outflows."
          },
          {
            "step": "Inference",
            "description": "Large inflows followed by immediate and frequent outflows suggest fund distribution to obscure origins."
          },
          {
            "step": "Conclusion",
            "description": "Evaluate whether the balance fluctuations are consistent with phishing node behavior."
          }
        ]
      },
      {
        "dimension": "Transaction Frequency and Timing",
        "steps": [
          {
            "step": "Observation",
            "description": "Review the node's transaction timeline, particularly its activity following large transfers."
          },
          {
            "step": "Inference",
            "description": "Frequent bursts of transactions after large inflows may indicate an attempt to distribute funds quickly and evade tracking."
          },
          {
            "step": "Conclusion",
            "description": "Assess whether these patterns align with malicious fund movement strategies."
          }
        ]
      },
      {
        "dimension": "Interaction with Other Nodes",
        "steps": [
          {
            "step": "Observation",
            "description": "Evaluate the node's connections with known phishing nodes, unknown entities, and its overall network interactions."
          },
          {
            "step": "Inference",
            "description": "High outgoing interactions with phishing or unknown nodes raise significant red flags, as they are indicative of malicious intent."
          },
          {
            "step": "Conclusion",
            "description": "Determine whether the node's connections suggest coordinated phishing activities."
          }
        ]
      },
      {
        "dimension": "Internal Transactions",
        "steps": [
          {
            "step": "Observation",
            "description": "Investigate the volume and frequency of internal transactions."
          },
          {
            "step": "Inference",
            "description": "Frequent low-value internal transactions often indicate self-circulation, a common tactic for obfuscating fund origins."
          },
          {
            "step": "Conclusion",
            "description": "Assess whether the internal transaction patterns are consistent with attempts to hide financial flows."
          }
        ]
      },
      {
        "dimension": "Final Classification and Analysis",
        "steps": [
          {
            "step": "Summary",
            "description": "Summarize the key findings from all dimensions, highlighting the most significant indicators of phishing or non-phishing behavior."
          },
          {
            "step": "Decision",
            "description": "Based on the summarized analysis, classify the node as either 'Phishing Node' or 'Non-Phishing Node'."
          },
          {
            "step": "Justification",
            "description": "Provide a concise justification for the classification, ensuring it aligns with the evidence presented in the analysis."
          }
        ]
      }
    ]
  },
  "output_format": {
    "response": "Provide a JSON-formatted analysis with no additional output.",
    "format": "json",
    "example": {
      "label": 0 or 1,
      "analysis": {
        ...
      }
    }
  }
}
""",
        "input": f""" 
{node_description}
""",
        "output": f"""
{output}
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


def process_cluster(index, dataset_name, delay, dataset_dir, queue, gpt_response):
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
        if node in gpt_response:  # 检查 node 是否在 analysis_dict 中
            analysis = gpt_response[node]["analysis"]  # 获取对应的 value
            result = process_node(node, add_histroy_train_graph, add_histroy_train_labels, analysis)
            queue.put(result)  # 将处理结果放入队列
    print(f"Cluster {index} processing completed, all nodes have been added to queue.")


def main():
    num_workers = 8  # 控制同时进行的任务数量
    dataset_name = "MulDiGraph"
    multiple_list = ["500"]
    delay = 5
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
        sorted_dataset_list = sorted_dataset_list[2:4]
        print(f"Dataset_list: {sorted_dataset_list}")

        result_dir = f'../../dataset/finetune/finetune_cot_3_json/'
        os.makedirs(result_dir, exist_ok=True)

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
            futures = []
            for i, index in enumerate(sorted_dataset_list):
                analysis_file_path = f"../../prompt/result/gpt-4o-mini/single-expert/{dataset_name}/{multiple}/delay_{delay}/sft_cot_data_by_label_json/{index}.xlsx"
                analysis_dict = get_cot_3_analysis(analysis_file_path)
                # 将任务提交到进程池
                future = executor.submit(
                process_cluster,
                index,
                dataset_name,
                delay,
                dataset_dir,
                queue_list[i],
                analysis_dict
                )
                futures.append(future)  # 将 future 存入列表

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
