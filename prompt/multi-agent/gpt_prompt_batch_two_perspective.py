import re
import json
import time
import os
import pandas as pd
from prompt.single_expert.describe_single_node_testing import describe_single_node_test
from prompt.single_expert.most_representative_sample_selection import generate_representative_details
from prompt.single_expert.head_recent_sample_selection import generate_train_node_detail, load_graph, load_labels
from openai import OpenAI, APIConnectionError, RateLimitError

API_ENDPOINT_ca = "https://api.openai-proxy.org/v1"
API_ENDPOINT_openai = "https://api.openai.com/v1"

test_type = "gpt-4o-mini"
model_gpt3 = "gpt-3.5-turbo"
model_gpt4 = "gpt-4"
model_gpt4mini = "gpt-4o-mini"
model_gpt4_32k = "gpt-4-32k"
model_gpt4o = "chatgpt-4o-latest"


def gpt_chat(messages, API=API5, model=model_gpt4mini, temperature=0, max_tokens=16384):
    client = OpenAI(
        base_url=API_ENDPOINT_openai,
        api_key=API,
    )
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response = chat_completion.choices[0].message.content
    return response


def expert_A_model(train_node_detail, test_node_detail, message_task_A, message_rule, message_attention_A,
                   messages_system_A):
    print("\nWait for llm expert ...\n")
    messages_input_A = f"""
[Examples]
The following are some example nodes, including both phishing nodes and normal nodes, provided for comparative learning.
Pay special attention to the **transaction details** of each node, including transaction frequency, transaction amounts, counterparties, and timing.
These are key factors that differentiate phishing nodes from normal nodes:
{train_node_detail}

[TASK]
I will provide you with a “Target Node” information, please pay attention to the information of this “Target Node”. 
Your task is to analyze and make a judgment on whether it is a phishing node:

[“Target Node”]
{test_node_detail}

[Evaluation]
{message_task_A}
    """
    messages_A = [
        {"role": "system", "content": messages_system_A},
        {"role": "user", "content": messages_input_A},
    ]
    response_A = gpt_chat(messages_A)
    return messages_A, response_A


def expert_B_model(train_node_detail, test_node_detail, message_task_B, message_rule, message_attention,
                   messages_system_B):
    print("\nWait for llm expert ...\n")
    messages_input = f"""
[Examples]
The following are some example nodes, including both phishing nodes and normal nodes, provided for comparative learning.
Pay special attention to the **transaction details** of each node, including transaction frequency, transaction amounts, counterparties, and timing.
These are key factors that differentiate phishing nodes from normal nodes:
{train_node_detail}

[TASK]
I will provide you with a “Target Node” information, please pay attention to the information of this “Target Node”. 
Your task is to analyze and make a judgment on whether it is a phishing node:

[“Target Node”]
{test_node_detail}

[Evaluation]
{message_task_B}
    """
    messages_B = [
        {"role": "system", "content": messages_system_B},
        {"role": "user", "content": messages_input},
    ]
    response_B = gpt_chat(messages_B)
    return messages_B, response_B


def expert_judge_model(test_node_detail, response_A, response_B, message_task_judge, message_rule,
                       message_attention, messages_system_judge):
    print("\nWait for llm expert D ...\n")
    messages_input = f"""    
[TASK]
I will provide you with a “Target Node” information, please pay attention to the information of this “Target Node” 
Your task is to analyze and make a judgment on whether it is a phishing node:

[“Target Node”]
{test_node_detail}

[Expert Analysis]
I will now provide you with the analysis results from two different experts, each of whom has evaluated Ethereum transaction nodes from a unique perspective.

Expert A believes that the target node is a phishing node involved in fraudulent activities. His analysis is as follows:
{response_A}

On the contrary, Expert B believes that the target node is not a phishing node involved in fraudulent activities. His analysis is as follows:
{response_B}

[Evaluation]
{message_task_judge}
{message_rule}
{message_attention}
    """
    messages = [
        {"role": "system", "content": messages_system_judge},
        {"role": "user", "content": messages_input},
    ]
    print(messages_input)
    response = gpt_chat(messages, API=API5, model=model_gpt4o)
    print(response)
    return messages, response


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
    response_B = gpt_chat(messages_B)
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
    response_C = gpt_chat(messages_C)
    return response_C


def main():
    # 对不同划分进行循环预测
    # strategy_list = ["head", "recent", "represent"]
    strategy = "recent"
    agent = "multi-agent"
    dataset_name = "ZipZap"
    multiple = "200"
    delay = 5
    dataset_dir = f'../../dataset/clustered_graphs/{dataset_name}/{multiple}/delay_{delay}'
    # if os.path.exists(dataset_dir):
    #     print(f"Dir exists: {dataset_dir}")
    #     dataset_list = os.listdir(dataset_dir)
    # else:
    #     print(f"Dir does not exists: {dataset_dir}")
    #
    # def natural_sort_key(filename):
    #     return [int(s) if s.isdigit() else s for s in re.split('([0-9]+)', filename)]
    #
    # sorted_dataset_list = sorted(dataset_list, key=natural_sort_key)
    # print(f"Dataset_list: {sorted_dataset_list}")

    sorted_dataset_list = ["3", "4", "5"]

    for index in sorted_dataset_list:
        print(f'\n *** Get Started -- cluster_{index} ***\n')
        result_dir = f'../result/{test_type}/{agent}/{dataset_name}/{multiple}/delay_{delay}/{strategy}_two_perspective'
        os.makedirs(result_dir, exist_ok=True)
        result_file_path = f'{result_dir}/{index}.xlsx'

        # Load 训练集 提示词
        train_node_detail = None
        if strategy == "recent":
            graph_train_file_path = f'{dataset_dir}/{index}/train.pkl'
            labels_train_file_path = f'{dataset_dir}/{index}/train_labels.json'
            train_node_detail = generate_train_node_detail(graph_train_file_path, labels_train_file_path)
        elif strategy == "head":
            graph_train_file_path = f'{dataset_dir}/1/train.pkl'
            labels_train_file_path = f'{dataset_dir}/1/train_labels.json'
            train_node_detail = generate_train_node_detail(graph_train_file_path, labels_train_file_path)
        elif strategy == "represent":
            subgraph_file_path_list = []
            labels_file_path_list = []
            for i in range(1, int(index) + 1):
                graph_train_file_path = f'{dataset_dir}/{str(i)}/train.pkl'
                labels_train_file_path = f'{dataset_dir}/{str(i)}/train_labels.json'
                subgraph_file_path_list.append(graph_train_file_path)
                labels_file_path_list.append(labels_train_file_path)
            train_node_detail = generate_representative_details(subgraph_file_path_list, labels_file_path_list, 2, 2)

        # 引入prompt
        prompts_file_path = "../prompt/MultiAgent_two_perspective.json"
        prompts = json.load(open(prompts_file_path, 'r'))
        message_attention = prompts['attention']
        message_rule = prompts['rules']

        # --------A------------
        # 专家A
        message_task_A = prompts['task_A']
        messages_system_A = prompts['system_A']

        # --------B------------
        # debater
        message_task_B = prompts['task_B']
        messages_system_B = prompts['system_B']

        # --------judge------------
        # debater
        message_task_judge = prompts['task_judge']
        messages_system_judge = prompts['system_judge']

        # Load 测试集 in 提示词
        graph_test_file_path = f'{dataset_dir}/{index}/test.pkl'
        labels_test_file_path = f'{dataset_dir}/{index}/test_labels.json'
        graph_test = load_graph(graph_test_file_path)
        labels_test = load_labels(labels_test_file_path)

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
            test_node_detail, is_phishing = describe_single_node_test(graph_test, labels_test, node)

            start_time = time.time()
            score = 0
            while True:
                try:
                    print(f"========================== Expert: A ==========================")
                    messages_A, response_A = expert_A_model(train_node_detail, test_node_detail, message_task_A,
                                                            message_rule,
                                                            message_attention, messages_system_A)
                    print("Expert A success !!!")
                    print(f"========================== Expert: B ==========================")
                    messages_B, response_B = expert_B_model(train_node_detail, test_node_detail, message_task_B,
                                                            message_rule,
                                                            message_attention, messages_system_B)
                    print("Expert debater success !!!")
                    print(f"========================== Expert: judge ==========================")
                    messages_judge, response_judge = expert_judge_model(test_node_detail, response_A, response_B,
                                                                        message_task_judge,
                                                                        message_rule, message_attention,
                                                                        messages_system_judge)
                    print("Expert judge success !!!")
                    score = output_score(response_judge)
                    print("output_score success !!!")
                    process_score_return = process_score(score)
                    if type(process_score_return) is int:
                        score = process_score_return
                    else:
                        score = check_score(response_judge, score, process_score_return)
                        if type(process_score(score)) is not int:
                            score = 0
                except (APIConnectionError, RateLimitError) as e:
                    print(f"An error occurred at node index {i + 1}: {e}. Retrying...")
                    time.sleep(300)
                    continue
                break
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Score: {str(score)}, Predict time： {execution_time} s")

            # 将结果追加到 DataFrame
            new_row = pd.DataFrame({
                'index': [i + 1],
                'node_address': [node],
                'is_phishing': [is_phishing],
                'score': [score],
                'messages': [messages_judge],
                'response': [response_judge],
            })

            if os.path.exists(result_file_path):
                with pd.ExcelWriter(result_file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                    if os.path.exists(result_file_path):
                        existing_rows = pd.read_excel(result_file_path).shape[0]
                    new_row.to_excel(writer, index=False, header=False, startrow=existing_rows + 1)
            else:
                df_results = pd.concat([df_results, new_row], ignore_index=True)
                df_results.to_excel(result_file_path, index=False)

            print(f"Node {str(i + 1)} result saved to: {result_file_path}")


if __name__ == '__main__':
    main()
