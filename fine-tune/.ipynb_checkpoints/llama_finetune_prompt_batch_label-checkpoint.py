import re
import json
import time
import os
import torch
import transformers
import pandas as pd
from describe_single_node_testing import describe_single_node_test
from add_history import *
from purest_sample_selection import generate_purest_detail
import argparse

parser = argparse.ArgumentParser(description='script for model.')
# 添加命令行参数
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--delay', type=int, required=True)
parser.add_argument('--strategy', type=str, required=True)
parser.add_argument('--test_dir', type=str, required=True)
parser.add_argument('--result_dir', type=str, required=True)
parser.add_argument('--index', type=int, required=True)
parser.add_argument('--multiple', type=int, required=True)

# 解析参数
args = parser.parse_args()

# 打印参数
print(f"model_path: {args.model_path}")
print(f"dataset_name: {args.dataset_name}")
print(f"delay: {args.delay}")
print(f"strategy: {args.strategy}")
print(f"test_dir: {args.test_dir}")
print(f"result_dir: {args.result_dir}")
print(f"index: {args.index}")
print(f"multiple: {args.multiple}")


test_type = "llama_finetune"
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
pipeline = transformers.pipeline(
    "text-generation",
    model=args.model_path,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "pad_token_id": tokenizer.eos_token_id
    },
    device_map="auto",
)


def llama_chat(messages):
    print(f"Input content len: {len(messages)}")
    if len(messages) > 80000:
        messages = messages[:80000]
    print(f"Truncate input content len: {len(messages)}")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    outputs = pipeline(
        messages,
        max_new_tokens=128000,
        max_time=3.5,
        do_sample=False,
        num_return_sequences=1
    )
    generated_text = outputs[0]["generated_text"]
    if generated_text.startswith(messages):
        generated_text = generated_text[len(messages):].strip()
    return generated_text


def single_expert_model(test_node_detail, message_task_A, message_rule, message_attention_A,
                        messages_system_A):
    print("\nWait for llm expert ...\n")
    messages = test_node_detail
    response = llama_chat(messages)
    return messages, response


def extract_first_non_repeating(output):
    # 使用正则表达式匹配第一个连续重复的子字符串
    match = re.match(r"(.*?)(\1+)?", output)
    if match:
        return match.group(1).strip()  # 返回第一个非重复子字符串并去除多余空格
    return output


def main():
    # multiple_list = ["500", "1000", "2000"]
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

    result_dir = f'{args.result_dir}/{test_type}/{agent}/{dataset_name}/{multiple}/delay_{delay}/{strategy}'  
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = f'{result_dir}/{index}.xlsx'
    print(f'\n *** Get Started -- multiple: {multiple}, cluster_{index} ***\n')
         
    # # Load 训练集 提示词
    # train_node_detail = None
    # print("================== add_histroy_graph for train ==================")
    # if strategy == "purest":
    #     graph_train_file_path = f'{test_dir}/{index}/train.pkl'
    #     labels_train_file_path = f'{test_dir}/{index}/train_labels.json'
    #     nodemap_train_file_path = f'{test_dir}/{index}/train_nodemap.json'
    #     graph_train = load_graph(graph_train_file_path)
    #     labels_train = load_json(labels_train_file_path)
    #     nodemap_train = load_json(nodemap_train_file_path)
    #     add_histroy_train_graph, add_histroy_train_labels = add_history(graph_all, graph_train, labels_train, nodemap_train,
    #                                                         train_start_date, delay)
    #     print(f"graph_train nodes: {len(graph_train)}")
    #     current_labels = load_json(f"/root/data/{dataset_name}/LLM/delay_{delay}/{index}/train_labels.json")
    #     train_node_detail = generate_purest_detail(add_histroy_train_graph, add_histroy_train_labels, graph_train, current_labels, nodemap_train)


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

    # 开始测试 保存测试结果
    # 创建一个空的 DataFrame
    df_results = pd.DataFrame(columns=['index', 'node_address', 'is_phishing', 'score', 'messages', 'response'])
    if os.path.exists(result_file_path):
        existing_df = pd.read_excel(result_file_path)
        existing_rows = len(existing_df)
    else:
        existing_rows = 0

    for i, node in enumerate(graph_test.nodes):
        if i < existing_rows:
            print(f"Node {str(i + 1)} already processed, skipping...")
            continue

        print(f"=================================== node index: {str(i + 1)} ===================================")
        test_node_detail, is_phishing = describe_single_node_test(add_histroy_graph, labels_test, node)

        start_time_A = time.time()
        messages, response = single_expert_model(test_node_detail, message_task_A,
                                                 message_rule, message_attention_A, messages_system_A)
        print("single_expert_model success !!!")
        print(f"response: {response}")
        if response.startswith("Non-Phishing"):
            score = 0
        elif response.startswith("Phishing"):
            score = 100
        else:
            print(f"error response: {response}")
            score = 0
        end_time_A = time.time()
        execution_time_A = end_time_A - start_time_A
        print(f"Score: {str(score)}, Predict time： {execution_time_A} s")

        # 将结果追加到 DataFrame
        new_row = pd.DataFrame({
            'index': [i + 1],
            'node_address': [node],
            'is_phishing': [is_phishing],
            'score': [score],
            'messages': [messages],
            'response': [response],
        })

        if os.path.exists(result_file_path):
            with pd.ExcelWriter(result_file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                if os.path.exists(result_file_path):
                    existing_rows = pd.read_excel(result_file_path).shape[0]  # 重新计算现有行数
                new_row.to_excel(writer, index=False, header=False, startrow=existing_rows + 1)
        else:
            df_results = pd.concat([df_results, new_row], ignore_index=True)
            df_results.to_excel(result_file_path, index=False)

        print(f"Node {str(i + 1)} result saved to: {result_file_path}")


if __name__ == '__main__':
    main()
