import random
import re
import json
import time
import os
import pandas as pd
import tiktoken

from describe_single_node_testing_token_4096 import describe_single_node_test
from head_recent_sample_selection import load_graph, load_json
from openai import OpenAI, APIConnectionError, BadRequestError, RateLimitError

from add_history import add_history



API_ENDPOINT_ca = "https://api.openai-proxy.org/v1"
API_ENDPOINT_openai = "https://api.openai.com/v1"

test_type = "gpt-4o-mini"
model_gpt3 = "gpt-3.5-turbo"
model_gpt4 = "gpt-4"
model_gpt4mini = "gpt-4o-mini"
model_gpt4_32k = "gpt-4-32k"

api_list = [API1, API2, API3, API4, API5, API6, API7, API8, API9, API10, API11, API12, API13, API14]
current_api_key = API14


def change_api_key():
    global current_api_key
    current_api_key = random.choice(api_list)
    pass


def gpt_chat(messages, model=model_gpt4mini, temperature=0, max_tokens=16384):
    encoding = tiktoken.encoding_for_model(model)
    token_list = encoding.encode(messages[1]["content"])
    print(f"Input tokens: {len(token_list)}")

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


def single_expert_model(test_node_detail, message_task_A, message_rule, message_attention_A,
                        messages_system_A, label):
    print("\nWait for llm expert ...\n")
    label = "Phishing" if label == 1 else "Non-phishing"
    messages_input_A = f"""
[TASK]
I will provide you with a “Target Node” information and the true label(Phishing or Non-phishing) of it. 
Your task is to analyze the information and tell me why this node is Phishing or Non-phishing.:

[“Target Node”]
{test_node_detail}

[Label]
{label}
    """
    messages_A = [
        {"role": "system", "content": messages_system_A},
        {"role": "user", "content": messages_input_A},
    ]
    response_A = gpt_chat(messages_A)
    return messages_A, response_A

def explain2analysis(response):
    system_content = ("You are an AI assistant tasked with extracting the node analysis, "
                      "and give a final confidence score based on the analysis. "
                      "Ignore all other irrelevant information or text, and format the output strictly in the following JSON format.")

    user_content = f"""
Below is an analysis of a node's attributes and behavior in the network.

[Node Analysis]
{response}

[TASK]
Your task is to extract the analysis and  give a confidence score of the analysis based on logic. Print **only the analysis, the confidence score** in the following JSON format:
{{
  "Analysis": <analysis>
  "Confidence": <confidence_score>
}}

[Rules]
Your confidence score should be between 0 and 100.
0-20: Very low confidence
21-40: Low confidence
41-60: Medium confidence
61-80: High confidence
81-100: Very high confidence
It should be based on:
- The logic and reasoning behind the analysis.
- The quality of the analysis.

[Output]
"""
    messages_B = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    response_B = gpt_chat(messages_B)
    return response_B



def process_score_json(score_str):
    try:
        # 定义正则表达式，匹配字符串
        pattern = r'{\s*"Analysis":\s*"(.*?)",\s*"Confidence":\s*(\d+)\s*}'

        # 使用正则表达式查找匹配项
        match = re.search(pattern, score_str)

        if match:
            # 提取分数值
            analysis = match.group(1)
            confidence_score = int(match.group(2))

            # 检查是否找到了 Node 和 Confidence 的分数
            if analysis is None or confidence_score is None:
                return "analysis or confidence score missing"

            # 验证分数是否在有效范围内
            if  0 <= confidence_score <= 100:
                result = {"analysis": analysis, "confidence_score": confidence_score}
                return result
            else:
                return f"One of the scores is out of range (0-100): Confidence Score={confidence_score}"
        else:
            return None, "No valid JSON pattern found"

    except json.JSONDecodeError:
        return "Invalid JSON format"
    except ValueError:
        return "Invalid score format"


def main():
    strategy = "sft_data_by_label"
    agent = "single-expert"
    dataset_name = "MulDiGraph"
    multiple = "100"
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
    sorted_dataset_list = ["15"]
    print(f"Dataset_list: {sorted_dataset_list}")
    for index in sorted_dataset_list:
        print(f'\n *** Get Started -- cluster_{index} ***\n')
        start_date_file_path = f'../../dataset/{dataset_name}/start_times.json'
        train_start_date = load_json(start_date_file_path)["train"][index]
        graph_all_file_path = f'../../dataset/{dataset_name}/data/{dataset_name}.pkl'
        graph_all = load_graph(graph_all_file_path)

        result_dir = f'../result/{test_type}/{agent}/{dataset_name}/{multiple}/delay_{delay}/{strategy}'
        os.makedirs(result_dir, exist_ok=True)
        result_file_path = f'{result_dir}/{index}.xlsx'

        # 引入prompt
        prompts_file_path = "../prompt/SingleExpert_confidence.json"
        prompts = json.load(open(prompts_file_path, 'r'))
        message_task_A = prompts['task']
        messages_system_A = prompts['system']
        message_attention_A = prompts['attention']
        message_rule = prompts['rules']

        print("================== add_histroy_graph for train ==================")
        graph_train_file_path = f'{dataset_dir}/{index}/train.pkl'
        labels_train_file_path = f'{dataset_dir}/{index}/train_labels.json'
        nodemap_train_file_path = f'{dataset_dir}/{index}/train_nodemap.json'
        graph_train = load_graph(graph_train_file_path)
        labels_train = load_json(labels_train_file_path)
        nodemap_train = load_json(nodemap_train_file_path)
        add_histroy_graph, labels_train = add_history(graph_all, graph_train, labels_train, nodemap_train,
                                                      train_start_date,
                                                      delay)
        print(f"graph_test nodes: {len(graph_train)}")
        # 开始测试 保存测试结果
        # 创建一个空的 DataFrame
        df_results = pd.DataFrame(
            columns=['index', 'node_address', 'is_phishing', 'confidence', 'analysis', 'node_detail'])
        if os.path.exists(result_file_path):
            existing_df = pd.read_excel(result_file_path)
            existing_rows = len(existing_df)
        else:
            existing_rows = 0

        for i, node in enumerate(graph_train.nodes):

            if i < existing_rows:
                print(f"Node {str(i + 1)} already processed, skipping...")
                continue

            print(f"=================================== node index: {str(i + 1)} ===================================")
            train_node_detail, is_phishing = describe_single_node_test(add_histroy_graph, labels_train, node)

            start_time_A = time.time()
            score = 0
            confidence = 0
            analysis = ""
            response = ""
            while True:
                try:
                    messages, response = single_expert_model(train_node_detail, message_task_A,
                                                             message_rule, message_attention_A, messages_system_A, is_phishing)
                    print("single_expert_model success !!!")
                    score_str = explain2analysis(response)
                    print("output_score success !!!")
                    result = process_score_json(score_str)
                    if isinstance(result, dict):
                        analysis = result["analysis"]
                        confidence = result["confidence_score"]
                    print(f"response: {response}")
                    print(f"analysis: {analysis}")
                    print(f"confidence: {confidence}")
                except (APIConnectionError, RateLimitError) as e:
                    print(f"An error occurred at node index {i + 1}: {e}. Retrying...")
                    change_api_key()
                    print(f"Change api key ...")
                    continue
                except BadRequestError as e:
                    print(f"BadRequestError occurred: {e}")
                    print(
                        f"len train_node_detail: {len(train_node_detail)}")
                    train_node_detail = train_node_detail[:len(train_node_detail) - 200]
                    continue
                break
            end_time_A = time.time()
            execution_time_A = end_time_A - start_time_A
            print(f"Confidence: {confidence},  Predict time： {execution_time_A} s")

            # 将结果追加到 DataFrame
            new_row = pd.DataFrame({
                'index': [i + 1],
                'node_address': [node],
                'is_phishing': [is_phishing],
                'confidence': [confidence],
                'analysis': [analysis],
                'node_detail': [train_node_detail],
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

