import re
import json
import time
import os
import torch
import transformers
import pandas as pd
from describe_single_node_testing_token_4096 import describe_single_node_test
from add_history import *
from purest_sample_selection import generate_purest_detail
import argparse
from torch.multiprocessing import Pool, set_start_method

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
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM  # 在函数内导入
    print(f"[GPU {gpu_id}] Initializing model...")

    # 加载 Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)

    # 使用 accelerate 自动加载模型，并启用 bfloat16 加速
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 使用 accelerate 自动映射设备
    )

    # 创建 pipeline，不要指定 device 参数
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    print(f"[GPU {gpu_id}] Model initialized successfully.")
    return pipeline, tokenizer, model
    

def get_output_label_confidence(pipeline, tokenizer, model, messages):
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    def count_tokens(text):
        tokens = tokenizer.encode(text, add_special_tokens=False)  # 对输入文本进行编码
        return len(tokens)
    print(f"count_tokens: {count_tokens(messages)}")
    
    # 生成模型输出
    outputs = pipeline(
        messages,
        max_new_tokens=128000,  # 限制生成的最大长度
        max_time=50,
        do_sample=False,
        num_return_sequences=1
    )

    # 提取生成的文本
    generated_text = outputs[0]["generated_text"]

    # 找到第二次出现的 "### Output Label"
    first_occurrence = generated_text.find("### Output Label")
    if first_occurrence == -1:
        raise ValueError("The first '### Output Label' not found.")

    second_occurrence = generated_text.find("### Output Label", first_occurrence + len("### Output Label"))
    if second_occurrence == -1:
        raise ValueError("The second '### Output Label' not found.")

    # 找到第二次出现后的第一个有效字符位置
    label_start = second_occurrence + len("### Output Label")

    # 跳过空格和换行符，找到第一个有效字符的位置
    while label_start < len(generated_text) and generated_text[label_start] in {'\n', ' ', '\t'}:
        label_start += 1

    if label_start >= len(generated_text):
        raise ValueError("No non-whitespace character found after the second '### Output Label'.")

    # 提取该字符
    first_char = generated_text[label_start]
    print(f"First Character after the second ### Output Label': '{first_char}'")

    # 将文本到该字符位置进行 token 化
    tokenized_text = tokenizer(generated_text[:label_start + 1], return_tensors="pt").to("cuda:0")

    # 获取模型的 logits
    with torch.no_grad():
        model_outputs = pipeline.model(**tokenizer(messages, return_tensors="pt").to("cuda:0"))
        logits = model_outputs.logits  # [batch_size, seq_len, vocab_size]
    
    # print(f"Logits shape: {logits.size()}, Token shape: {tokenized_text.input_ids.size()}")

    # 获取 token 数量，并确保与 logits 对齐
    num_tokens = tokenized_text.input_ids.size(1)

    # 如果 token 数量超出 logits 长度，进行对齐处理
    if num_tokens > logits.size(1):
        token_index = logits.size(1) - 1  # 使用 logits 中的最后一个位置
    else:
        token_index = num_tokens - 1  # 正常情况下，使用最后一个 token 的 logits

    # 获取该字符的 token ID
    char_token_id = tokenizer.encode(first_char, add_special_tokens=False)[0]

    # 使用 softmax 计算概率分布
    probs = torch.nn.functional.softmax(logits[0, token_index], dim=-1)

    # 获取该字符的置信度
    confidence_score = probs[char_token_id].item()

    return generated_text, first_char, confidence_score

    

# 每个 GPU 执行推理任务
def process_node_on_gpu(gpu_id, nodes, add_histroy_graph, labels_test):
    pipeline, tokenizer, model = initialize_model(gpu_id)
    results = []

    for node in nodes:
        # print(f"Processing node on GPU {gpu_id}: {node}")
        test_node_detail, is_phishing = describe_single_node_test(add_histroy_graph, labels_test, node)
        if len(test_node_detail) > 12000:
            test_node_detail = test_node_detail[:12000]
        print(f"len test_node_detail: {len(test_node_detail)}")
        
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
Accurately identifying phishing nodes is critical to protecting the financial ecosystem. Use the provided data to determine whether the node exhibits phishing behavior. Provide a concise analysis and ends with a clear, definitive answer in this format:  
### Output Label
(0 or 1)
### Analysis
"The analysis of Node ..." or "The 'Target Node' exhibits ..."

### Target Node Information  
{test_node_detail}

### Output Label
"""
        start_time_A = time.time()
        generated_text, first_char, confidence_score = get_output_label_confidence(pipeline, tokenizer, model, messages)
        if first_char == "0":
            label = 0
        elif first_char == "1":
            label = 1
        else:
            print(f"error first_char: {first_char}")
            label = 0
            confidence_score = 0
        end_time_A = time.time()
        execution_time_A = end_time_A - start_time_A
        print(f"[GPU {gpu_id}] Node {node} processed.")
        print(f"first_char: {first_char}, label: {label}, confidence_score: {confidence_score},  Predict time： {execution_time_A} s")
        results.append((node, is_phishing, label, generated_text, first_char, confidence_score))
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
    df_results = pd.DataFrame(columns=['index', 'node_address', 'is_phishing', 'label', 'confidence_score', 'response'])
    for gpu_results in all_results:
        for i, (node, is_phishing, label, generated_text, first_char, confidence_score) in enumerate(gpu_results):
            new_row = pd.DataFrame({
                'index': [i + 1],
                'node_address': [node],
                'is_phishing': [is_phishing],
                'label': [label],
                'confidence_score': [confidence_score],
                'response': [generated_text],
            })
            df_results = pd.concat([df_results, new_row], ignore_index=True)

    # 保存结果
    df_results.to_excel(result_file_path, index=False)
    print(f"Results saved to: {result_file_path}")

if __name__ == '__main__':
    main()
