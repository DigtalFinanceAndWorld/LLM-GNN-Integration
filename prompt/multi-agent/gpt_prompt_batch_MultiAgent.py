import random
import re
import json
import time
import os
import pandas as pd
import tiktoken

from prompt.single_expert.add_history import add_history
from prompt.single_expert.describe_single_node_testing import describe_single_node_test
from prompt.sample_select.representative_sample_selection import generate_representative_details
from prompt.sample_select.head_recent_sample_selection import generate_train_node_detail, load_graph, load_json
from prompt.sample_select.purest_sample_selection import generate_purest_detail
from openai import OpenAI, APIConnectionError, RateLimitError, BadRequestError


API1 = ""
API2 = ""
API3 = ""

API_ENDPOINT_ca = "https://api.openai-proxy.org/v1"
API_ENDPOINT_openai = "https://api.openai.com/v1"

test_type = "gpt-4o-mini"
model_gpt3 = "gpt-3.5-turbo"
model_gpt4 = "gpt-4"
model_gpt4mini = "gpt-4o-mini"
model_gpt4_32k = "gpt-4-32k"

api_list = [API1, API2, API3]
current_api_key = API1


def change_api_key():
    global current_api_key
    current_api_key = random.choice(api_list)
    pass


def gpt_chat(messages, model=model_gpt4mini, temperature=0, max_tokens=16384):
    encoding = tiktoken.encoding_for_model(model)
    token_list = encoding.encode(messages[1]["content"])
    print(f"Input tokens: {len(token_list)}, max_tokens: {max_tokens}")

    client = OpenAI(
        base_url=API_ENDPOINT_openai,
        api_key=current_api_key,
    )
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response = chat_completion.choices[0].message.content
    return response


def single_expert_model(train_node_detail, new_node_detail, message_task_A, message_rule, message_attention_A, messages_system_A):
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
{new_node_detail}

[Evaluation]
{message_task_A}
{message_rule}
{message_attention_A}
    """
    messages_A = [
        {"role": "system", "content": messages_system_A},
        {"role": "user", "content": messages_input_A},
    ]
    response_A = gpt_chat(messages_A)
    return messages_A, response_A


def multi_agent_model(new_node_detail, response_A, response_B, response_C, message_task_D, message_rule,
                      message_attention, messages_system_D):
    print("\nWait for llm expert D ...\n")
    messages_input = f"""    
[TASK]
I will provide you with a “Target Node” information, please pay attention to the information of this “Target Node” 
Your task is to analyze and make a judgment on whether it is a phishing node:

[“Target Node”]
{new_node_detail}

[Expert Analysis]
I will now provide you with the analysis results from three different experts, each of whom has evaluated Ethereum transaction nodes from a unique perspective. 
Expert A focuses on micro-level transaction behavior, analyzing the frequency of transactions, the distribution of transaction amounts, and any irregularities in transaction timing. 
Expert A analysis result: {response_A}

Expert B assesses network structure and node relationships, examining the interactions between nodes, their position in the network, and connections with known high-risk nodes. 
Expert B analysis result: {response_B}

Expert C provides a macro-level analysis of transaction patterns and attributes, evaluating overall transaction volumes, consistency of transaction behaviors, and comparisons with other nodes in the network. 
Expert C analysis result: {response_C}

[Evaluation]
{message_task_D}
{message_rule}
{message_attention}
    """
    messages = [
        {"role": "system", "content": messages_system_D},
        {"role": "user", "content": messages_input},
    ]
    response = gpt_chat(messages)
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
    # strategy_list = ["head", "recent", "represent", "history_recent", "purest"]
    strategy = "purest"
    agent = "multi-agent"
    dataset_name = "MulDiGraph"
    multiple = "500"
    delay = 5
    dataset_dir = f'../../dataset/clustered_graphs/{dataset_name}/{multiple}/delay_{delay}'
    if os.path.exists(dataset_dir):
        print(f"Dir exists: {dataset_dir}")
        dataset_list = os.listdir(dataset_dir)
    else:
        print(f"Dir does not exists: {dataset_dir}")

    def natural_sort_key(filename):
        return [int(s) if s.isdigit() else s for s in re.split('([0-9]+)', filename)]

    sorted_dataset_list = sorted(dataset_list, key=natural_sort_key)
    print(f"Dataset_list: {sorted_dataset_list}")

    for index in sorted_dataset_list:
        print(f'\n *** Get Started -- cluster_{index} ***\n')
        start_date_file_path = f'../../dataset/{dataset_name}/start_times.json'
        train_start_date = load_json(start_date_file_path)["train"][index]
        test_start_date = load_json(start_date_file_path)["test"][index]
        graph_all_file_path = f'../../dataset/{dataset_name}/data/{dataset_name}.pkl'
        graph_all = load_graph(graph_all_file_path)

        result_dir = f'../result/{test_type}/{agent}/{dataset_name}/{multiple}/delay_{delay}/{strategy}'
        os.makedirs(result_dir, exist_ok=True)
        result_file_path = f'{result_dir}/{index}.xlsx'

        # Load 训练集 提示词
        train_node_detail = None
        print("================== add_histroy_graph for train ==================")
        if strategy == "history_recent":
            graph_train_file_path = f'{dataset_dir}/{index}/train.pkl'
            labels_train_file_path = f'{dataset_dir}/{index}/train_labels.json'
            nodemap_train_file_path = f'{dataset_dir}/{index}/train_nodemap.json'
            graph_train = load_graph(graph_train_file_path)
            labels_train = load_json(labels_train_file_path)
            nodemap_train = load_json(nodemap_train_file_path)
            add_histroy_graph, labels_train = add_history(graph_all, graph_train, labels_train, nodemap_train,
                                                          train_start_date, delay)
            print(f"graph_train nodes: {len(graph_train)}")
            train_node_detail = generate_train_node_detail(add_histroy_graph, labels_train)
        elif strategy == "purest":
            graph_train_file_path = f'{dataset_dir}/{index}/train.pkl'
            labels_train_file_path = f'{dataset_dir}/{index}/train_labels.json'
            nodemap_train_file_path = f'{dataset_dir}/{index}/train_nodemap.json'
            graph_train = load_graph(graph_train_file_path)
            labels_train = load_json(labels_train_file_path)
            nodemap_train = load_json(nodemap_train_file_path)
            add_histroy_train_graph, add_histroy_train_labels = add_history(graph_all, graph_train, labels_train,
                                                                            nodemap_train,
                                                                            train_start_date, delay)
            print(f"graph_train nodes: {len(graph_train)}")
            current_labels = load_json(f"../../dataset/{dataset_name}/data/LLM/delay_{delay}/{index}/train_labels.json")
            train_node_detail = generate_purest_detail(add_histroy_train_graph, add_histroy_train_labels, graph_train,
                                                       current_labels, nodemap_train)
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
        prompts_file_path = "../prompt/MultiAgent_2.json"
        prompts = json.load(open(prompts_file_path, 'r'))
        message_attention = prompts['attention']
        message_rule = prompts['rules']

        # --------A------------
        # 专家A：微观交易行为分析专家
        message_task_A = prompts['task_A']
        messages_system_A = prompts['system_A']

        # --------B------------
        # 专家B：网络结构和节点关系分析专家
        message_task_B = prompts['task_B']
        messages_system_B = prompts['system_B']

        # --------C------------
        # 专家C：宏观交易模式与属性分析专家
        message_task_C = prompts['task_C']
        messages_system_C = prompts['system_C']

        # --------D------------
        # 专家D：总结分析专家
        message_task_D = prompts['task_D']
        messages_system_D = prompts['system_D']

        print("================== add_histroy_graph for test ==================")
        graph_test_file_path = f'{dataset_dir}/{index}/test.pkl'
        labels_test_file_path = f'{dataset_dir}/{index}/test_labels.json'
        nodemap_test_file_path = f'{dataset_dir}/{index}/test_nodemap.json'
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
            test_node_detail, is_phishing, _ = describe_single_node_test(add_histroy_graph, labels_test, node)

            start_time = time.time()
            score = 0
            while True:
                try:
                    print(f"========================== Expert: A ==========================")
                    messages_A, response_A = single_expert_model(train_node_detail, test_node_detail, message_task_A, message_rule,
                                                                 message_attention, messages_system_A)
                    print("Expert A success !!!")
                    print(f"========================== Expert: B ==========================")
                    messages_B, response_B = single_expert_model(train_node_detail, test_node_detail, message_task_B, message_rule,
                                                                 message_attention, messages_system_B)
                    print("Expert B success !!!")
                    print(f"========================== Expert: C ==========================")
                    messages_C, response_C = single_expert_model(train_node_detail, test_node_detail, message_task_C, message_rule,
                                                                 message_attention, messages_system_C)
                    print("Expert C success !!!")
                    print(f"========================== Expert: D ==========================")
                    messages_D, response_D = multi_agent_model(test_node_detail, response_A, response_B, response_C, message_task_D,
                                                               message_rule, message_attention, messages_system_D)
                    print("Expert D success !!!")
                    score = output_score(response_D)
                    print("output_score success !!!")
                    process_score_return = process_score(score)
                    if type(process_score_return) is int:
                        score = process_score_return
                    else:
                        score = check_score(response_D, score, process_score_return)
                        if type(process_score(score)) is not int:
                            score = 0
                except (APIConnectionError, RateLimitError) as e:
                    print(f"An error occurred at node index {i + 1}: {e}. Retrying...")
                    change_api_key()
                    print(f"Change api key ...")
                    continue
                except BadRequestError as e:
                    print(f"BadRequestError occurred: {e}")
                    print(
                        f"len train_node_detail: {len(train_node_detail)}, len test_node_detail: {len(test_node_detail)}")
                    train_node_detail = train_node_detail[:len(train_node_detail) - 1000]
                    test_node_detail = test_node_detail[:len(test_node_detail) - 200]
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
                'messages': [messages_D],
                'response': [response_D],
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
