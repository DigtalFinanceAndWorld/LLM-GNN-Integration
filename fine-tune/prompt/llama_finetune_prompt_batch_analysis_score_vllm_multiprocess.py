import re
import json
import time
import os
import torch
import transformers
import pandas as pd
from prompt.single_expert.add_history import add_history
from prompt.single_expert.describe_single_node_testing_token_4096 import *
import argparse
from torch.multiprocessing import Pool, set_start_method
from vllm import LLM, SamplingParams

# 确保在多 GPU 场景中使用 spawn 方法来初始化多进程
try:
    set_start_method('spawn')
except RuntimeError:
    pass

parser = argparse.ArgumentParser(description='Distributed model inference script.')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--delay', type=int, required=True)
parser.add_argument('--strategy', type=str, required=True)
parser.add_argument('--test_dir', type=str, required=True)
parser.add_argument('--result_dir', type=str, required=True)
parser.add_argument('--index', type=int, required=True)
parser.add_argument('--multiple', type=int, required=True)
parser.add_argument('--gpu_count', type=int, required=True)

args = parser.parse_args()

print(f"model_path: {args.model_path}")
print(f"dataset_name: {args.dataset_name}")
print(f"delay: {args.delay}")
print(f"strategy: {args.strategy}")
print(f"test_dir: {args.test_dir}")
print(f"result_dir: {args.result_dir}")
print(f"index: {args.index}")
print(f"multiple: {args.multiple}")
print(f"gpu_count: {args.gpu_count}")


def initialize_model(gpu_id):
    from vllm import LLM
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] Initializing model...")
    llm = LLM(
        model=args.model_path,
        max_model_len=4096
    )
    print(f"[GPU {gpu_id}] Model initialized successfully.")
    return llm
    

def get_output_label_confidence_vllm(llm, messages):
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # 配置采样参数
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.0
    )

    # 生成输出
    output = llm.generate(messages, sampling_params)

    # 提取生成的文本
    generated_text = output[0].outputs[0].text
    print(generated_text)

    try:
        # 找到第一次出现 "### Output" 的位置
        first_occurrence = generated_text.find("### Output")
        if first_occurrence == -1:
            raise ValueError("First '### Output' not found.")

        # 提取 "### Output" 后面的内容并解析为字典
        output_section = generated_text[first_occurrence + len("### Output"):].strip()

        # 将字符串解析为字典
        result = json.loads(output_section)

        # 检查字典是否包含 "label" 和 "confidence_score"
        if "Fraudulent Probability" not in result:
            raise ValueError("Missing 'Fraudulent Probability' in the parsed output.")

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Error parsing generated output: {e}")
        return generated_text, 50

    # 返回生成的文本、标签和置信度分数
    return generated_text, result["Fraudulent Probability"]
    

# 每个 GPU 执行推理任务
def process_node_on_gpu(gpu_id, nodes, add_histroy_graph, labels_test):
    llm = initialize_model(gpu_id)
    results = []

    for node in nodes:
        # print(f"Processing node on GPU {gpu_id}: {node}")
        test_node_detail, is_phishing = describe_single_node_test(add_histroy_graph, labels_test, node)
        if len(test_node_detail) > 12000:
            test_node_detail = test_node_detail[:12000]
        print(f"len test_node_detail: {len(test_node_detail)}")
        
        messages = f"""
### Instruction  
You are a blockchain analyst specializing in identifying fraudulent behavior within the Ethereum network. Your task is to analyze the transaction data of a Target Node and evaluate its likelihood of being a Phishing Node. This is a binary classification problem, where you will assess a Fraudulent Probability score, indicating the probability that the node is involved in phishing activities. 

### Task Description  
Evaluate the Target Node’s behavior based on these key attributes:  
1. **Node Degree**: The number of incoming and outgoing transactions. Suspicious nodes may exhibit unusually high outgoing transactions with low or zero-value amounts.  
2. **Transaction Amounts**: Look for unusual average transaction amounts, both incoming and outgoing, including internal transactions.  
3. **Transaction Frequency**: Check for high frequencies or sudden bursts of transactions, which could indicate malicious activity.  
4. **Counterparty Interaction**: Identify frequent interactions with unknown nodes, suspicious addresses, or a large number of zero-value transactions.  
5. **Internal Transactions**: Look for frequent internal transfers with no clear financial purpose, which could signal deceptive behavior.  
6. **Transaction Timeline**: Determine whether transactions are spread evenly over time or clustered in bursts, which might indicate phishing.  

### Objective
Accurately evaluating phishing nodes is critical to protecting the financial ecosystem. Use the provided data to determine the likelihood that the node exhibits phishing behavior. In addition to the evaluation, provide a Fraudulent Probability score (0–100), reflecting how strongly the observed patterns support your assessment. Provide a concise analysis that ends with a clear, definitive answer in this format:
### Analysis
"The analysis of Node ..." or "The 'Target Node' exhibits ..."
### Output
{{"Fraudulent Probability": ? (0-100)}}

### Target Node Information
{test_node_detail}

### Analysis
"""
        start_time_A = time.time()
        generated_text, score = get_output_label_confidence_vllm(llm, messages)
        if not 0 <= score <= 100:
            print(f"error score: {score}")
            score = 0
        end_time_A = time.time()
        execution_time_A = end_time_A - start_time_A
        print(f"[GPU {gpu_id}] Node {node} processed.")
        print(f"score: {score},  Predict time： {execution_time_A} s")
        results.append((node, is_phishing, generated_text, score))
    return results


def extract_first_non_repeating(output):
    # 使用正则表达式匹配第一个连续重复的子字符串
    match = re.match(r"(.*?)(\1+)?", output)
    if match:
        return match.group(1).strip()  # 返回第一个非重复子字符串并去除多余空格
    return output


# 分配节点到多个 GPU
def distribute_nodes_across_gpus(nodes, gpu_count):
    chunk_size = (len(nodes) + gpu_count - 1) // gpu_count
    print(f"[INFO] Distributed {len(nodes)} nodes across {gpu_count} GPUs. Chunk size: {chunk_size}")
    return [nodes[i * chunk_size: (i + 1) * chunk_size] for i in range(gpu_count)]

def main():
    agent = "single-expert"
    strategy = args.strategy
    dataset_name = args.dataset_name
    delay = args.delay
    index = str(args.index)
    multiple = args.multiple

    test_dir = f'{args.test_dir}/{dataset_name}/{multiple}/delay_{delay}'
    start_date_file_path = f'/root/data/{dataset_name}/start_times.json'
    train_start_date = load_json(start_date_file_path)["train"][index]
    test_start_date = load_json(start_date_file_path)["test"][index]
    graph_all_file_path = f'/root/data/{dataset_name}/{dataset_name}.pkl'
    graph_all = load_graph(graph_all_file_path)

    result_dir = f'{args.result_dir}/{agent}/{dataset_name}/{multiple}/delay_{delay}/{strategy}'  
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = f'{result_dir}/{index}.xlsx'
    print(f'\n *** Get Started -- multiple: {multiple}, cluster_{index} ***\n')

    # 引入prompt
    prompts_file_path = "/root/prompt/prompt/SingleExpert.json"
    prompts = json.load(open(prompts_file_path, 'r'))
    message_task_A = prompts['task']
    messages_system_A = prompts['system']
    message_attention_A = prompts['attention']
    message_rule = prompts['rules']

    # Load 测试集 in 提示词
    print("================== add_histroy_graph for test ==================")
    graph_test_file_path = f'{test_dir}/{index}/test.pkl'
    labels_test_file_path = f'{test_dir}/{index}/test_labels.json'
    nodemap_test_file_path = f'{test_dir}/{index}/test_nodemap.json'
    graph_test = load_graph(graph_test_file_path)
    labels_test = load_json(labels_test_file_path)
    nodemap_test = load_json(nodemap_test_file_path)
    add_histroy_graph, labels_test = add_history(graph_all, graph_test, labels_test, nodemap_test, test_start_date,
                                                 delay)
    print(f"graph_test nodes: {len(graph_test)}")
    nodes = list(graph_test.nodes)

    # 分配节点给各 GPU
    node_chunks = distribute_nodes_across_gpus(nodes, args.gpu_count)

    # 多 GPU 推理
    with Pool(processes=args.gpu_count) as pool:
        all_results = pool.starmap(
            process_node_on_gpu,
            [(gpu_id, node_chunks[gpu_id], add_histroy_graph, labels_test)
             for gpu_id in range(args.gpu_count)]
        )

    # 汇总所有 GPU 的结果
    df_results = pd.DataFrame(columns=['index', 'node_address', 'is_phishing', 'score', 'response'])
    for gpu_results in all_results:
        for i, (node, is_phishing, generated_text, score) in enumerate(gpu_results):
            new_row = pd.DataFrame({
                'index': [i + 1],
                'node_address': [node],
                'is_phishing': [is_phishing],
                'score': [score],
                'response': [generated_text]
            })
            df_results = pd.concat([df_results, new_row], ignore_index=True)

    # 保存结果
    df_results.to_excel(result_file_path, index=False)
    print(f"Results saved to: {result_file_path}")

if __name__ == '__main__':
    main()
