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
    content = messages[1]["content"]
    print(f"Input content len: {len(content)}")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    outputs = pipeline(
        messages,
        max_new_tokens=128000,
        max_time=200
    )
    return outputs[0]["generated_text"][-1]['content']
    

def single_expert_model(test_node_detail, message_task_A, message_rule, message_attention_A,
                        messages_system_A):
    print("\nWait for llm expert ...\n")
    messages_input_A = f"""
[TASK]
I will provide you with a “Target Node” information, please pay attention to the information of this “Target Node”. 
Your task is to analyze and make a judgment on whether it is a phishing node:

[“Target Node”]
{test_node_detail}

[Evaluation]
{message_task_A}
{message_rule}
{message_attention_A}
    """
    messages_A = [
        {"role": "system", "content": messages_system_A},
        {"role": "user", "content": messages_input_A},
    ]
    response_A = llama_chat(messages_A)
    return messages_A, response_A


def output_score(response_A):
    system_content = ("You are an AI assistant that extracts the final evaluation score from detailed node analysis. "
                      "Your task is to identify and extract only the numerical score from the analysis without any "
                      "additional text or explanation.")

    user_content = f"""
Below is an analysis of a node's attributes and behavior in the network, followed by a final evaluation score. Your task is to extract and print **only the final evaluation score**. Ignore all other information and text.

[Node Analysis]
{response_A}

[Output]
"""
    messages_B = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    response_B = llama_chat(messages_B)
    return response_B


def process_score(score):
    try:
        score_str = re.findall(r'\d+', str(score))
        if len(score_str) == 1:
            score_int = int(score_str[0])
            if 0 <= score_int <= 100:
                return score_int
            else:
                return f"Score {score_int} is out of range (0-100)"
        else:
            return "The output score is greater than one"
    except ValueError:
        return "Invalid score format"


def check_score(response_A, response_score, wrong_message):
    system_content = ("You are an AI assistant that extracts the final evaluation score from detailed node analysis. "
                      "Your task is to identify and extract only the numerical score from the analysis without any "
                      "additional text or explanation.")

    user_content = f"""
Below is an analysis of a node's attributes and behavior in the network, followed by a final evaluation score. Your task is to extract and print **only one the final evaluation score**. You only need to output a number between 0 and 100. Please must not output other information. Thank you!

[Node Analysis]
{response_A}

Wrong Output: {response_score}, Reason: {wrong_message}

The extracted evaluation score appears to be incorrect or out of the valid range (0-100). Could you please recheck and extract the **only one the final evaluation score** again, ensuring that it falls within the range of 0 to 100? If necessary, review the [Node Analysis] section to locate the correct score. You only need to output a number between 0 and 100. Please must not output other information. Thank you!
[Correct Output]
"""
    messages_C = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return response_C


def main():
    agent = "single-expert"
    # multiple_list = ["500", "1000", "2000"]
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
        score = output_score(response)
        print("output_score success !!!")
        process_score_return = process_score(score)
        if type(process_score_return) is int:
            score = process_score_return
        else:
            score = check_score(response, score, process_score_return)
            if type(process_score(score)) is not int:
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
