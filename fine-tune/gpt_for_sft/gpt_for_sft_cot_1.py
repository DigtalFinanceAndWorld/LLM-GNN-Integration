import random
import re
import json
import time
import os
import pandas as pd
import tiktoken

from prompt.single_expert.add_history import add_history
from prompt.single_expert.describe_single_node_testing_token_4096 import *
from openai import OpenAI, APIConnectionError, BadRequestError, RateLimitError


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
### Instruction  
You are a blockchain analyst specializing in identifying fraudulent behavior within the Ethereum network. Your task is to analyze the transaction data of a **Target Node** and classify it as either a **Phishing Node** or **Non-Phishing Node**. This is a **binary classification problem**, requiring you to return one of the following:  
- **1**: Phishing Node  
- **0**: Non-Phishing Node  


### Task Description  
Evaluate the Target Node’s behavior based on these key attributes. Remember that phishing nodes are rare and usually exhibit multiple suspicious patterns together. If the Target Node only partially matches phishing characteristics, it is more likely to be a Non-Phishing Node. 

Positive Features (Phishing Indicators): Describe any patterns or behaviors indicative of phishing, such as:
- High frequency of outgoing transactions, particularly with low or zero-value amounts.
- Unusual average transaction amounts that deviate significantly from normal usage.
- Frequent interactions with unknown nodes, high number of zero-value transactions, or sudden transaction bursts.

Negative Features (Non-Phishing Indicators): Identify patterns that indicate typical, non-suspicious behavior, such as:
- Transaction volumes and frequencies within normal ranges.
- Counterparty interactions primarily with known, non-phishing nodes.
- A steady transaction timeline without unusual bursts.


### Objective
Accurately identifying phishing nodes is critical to protecting the financial ecosystem. Use the provided data to identify **positive features** (indicative of phishing) and **negative features** (indicative of non-phishing) for the Target Node. Then, provide a **final evaluation** of whether the node exhibits phishing behavior. In addition to the label, provide a confidence score (50–100), reflecting how strongly the observed patterns support your classification. Provide a concise analysis that includes positive and negative features, and ends with a clear, definitive answer in this format:


### Analysis
**Positive Features**: List features that support phishing behavior.  
**Negative Features**: List features that counter phishing behavior.  
**Final Evaluation**: "The 'Target Node' exhibits ... based on the above characteristics."
### Output
{{"label": 0 or 1, "confidence_score": ? (50-100)}}


### Target Node Information  
{test_node_detail}


### Label Classification Rules
A node should only be labeled as a Phishing Node (1) if it exhibits multiple, strong indicators of phishing behavior across several dimensions (e.g., high transaction frequency, interactions with unknown addresses, unusual transaction amounts). If only one or two indicators are present, it is more likely to be a Non-Phishing Node (0). 


### Confidence Score Rules
The confidence score should be based on:
- **Clarity and Consistency in Reasoning**: Assess how clearly and logically the analysis is presented. The analysis should follow a coherent flow, with each point connecting logically to the next. High clarity and internal consistency should yield higher confidence scores.
- **Analytical Depth and Detail**: Evaluate the level of detail in identifying both positive and negative features. Analyses that include specific, relevant observations and address multiple dimensions of the node’s behavior (e.g., transaction frequency, counterparties, transaction amounts) should score higher in confidence.
- **Relevance and Completeness of Data**: Review the breadth and relevance of the data utilized in the analysis. An analysis that effectively uses available data, identifies unique patterns, and avoids unnecessary assumptions should receive a higher score. In cases of missing or sparse data, confidence should be lowered.
- **Certainty and Specificity of Conclusions**: Consider how definitive and specific the final conclusion is. Analyses that reach a conclusive determination with well-supported arguments should receive higher scores. For analyses with ambiguous or tentative conclusions, reduce the confidence score to reflect uncertainty.

Scoring Guide:
- **50-60**: Medium confidence. 
  - Scores closer to 50: Analysis is mostly clear but has notable gaps in logic or depth.
  - Scores closer to 55: Analysis is reasonably logical but misses multiple data points or lacks detail in certain areas.
  - Scores at 60: The analysis has a basic logical structure but requires more comprehensive data or depth for stronger support.

- **61-70**: Moderately high confidence.
  - Scores around 63: Mostly clear, logical flow but some inconsistencies in data relevance or detail.
  - Scores around 68-70: Few inconsistencies, logical structure with reasonable depth, but minor uncertainties may be present.

- **71-80**: High confidence.
  - Scores closer to 71-73: Analysis is generally clear and structured, with minimal data gaps.
  - Scores closer to 77-80: Very clear, detailed, and mostly complete, with just minor uncertainties.

- **81-90**: Very high confidence.
  - Scores around 81-85: Analysis is thorough with excellent clarity, minor uncertainties in data usage.
  - Scores around 86-90: Almost no uncertainties, highly relevant data, leading to a well-supported conclusion.

- **91-100**: Exceptional confidence.
  - Scores closer to 91-95: Comprehensive and highly reliable but with very slight room for additional detail.
  - Scores closer to 96-100: Analysis is extremely clear, fully detailed, and conclusive without any notable uncertainties.

If the analysis:
- Has unaddressed assumptions or lacks specific examples, reduce the score by 3-5 points within its range.
- Misses addressing certain aspects (e.g., frequency, transaction amount) or has inconsistent logic, reduce score by 2-3 points.


### Analysis
"""
    messages_A = [
        {"role": "system", "content": messages_system_A},
        {"role": "user", "content": messages_input_A},
    ]
    response_A = gpt_chat(messages_A)
    return messages_A, response_A

def explain2analysis(response):
    system_content = ("You are an AI assistant tasked with extracting key elements from the node analysis, "
                      "including positive features, negative features, the final conclusion, label, and a confidence score. "
                      "Ignore all irrelevant information or extraneous text, and format the output strictly in the following JSON format.")

    user_content = f"""
Below is an analysis of a node's attributes and behavior in the network.

[Node Analysis]
{response}

[TASK]
Your task is to extract the **positive features**, **negative features**, **conclusion**, **label**, and **confidence score** based on the logic provided. Print **only these elements** in the following JSON format:
{{
  "Positive_Features": <positive_features>,
  "Negative_Features": <negative_features>,
  "Conclusion": <conclusion>,
  "Label": <label>,
  "Confidence": <confidence_score>
}}

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
        # 尝试使用 JSON 解析器直接解析字符串
        parsed_data = json.loads(score_str.strip())

        # 检查必须字段是否存在
        required_fields = ["Positive_Features", "Negative_Features", "Conclusion", "Label", "Confidence"]
        for field in required_fields:
            if field not in parsed_data:
                return f"Missing required field: {field}"

        # 获取字段值
        positive_features = parsed_data["Positive_Features"]
        negative_features = parsed_data["Negative_Features"]
        conclusion = parsed_data["Conclusion"]
        label = parsed_data["Label"]
        confidence_score = parsed_data["Confidence"]
        
        # 将 Positive_Features 和 Negative_Features 列表转换为分点字符串格式
        if isinstance(positive_features, list):
            positive_features = "\n- " + "\n- ".join(positive_features)
        else:
            positive_features = str(positive_features)

        if isinstance(negative_features, list):
            negative_features = "\n- " + "\n- ".join(negative_features)
        else:
            negative_features = str(negative_features)

        # 验证分数和标签是否在有效范围内
        if not (0 <= confidence_score <= 100):
            return f"Confidence Score is out of range (0-100): {confidence_score}"
        if label not in [0, 1]:
            return f"Label is invalid, expected 0 or 1, got: {label}"

        # 返回结构化结果
        result = {
            "positive_features": positive_features,
            "negative_features": negative_features,
            "conclusion": conclusion,
            "label": label,
            "confidence_score": confidence_score
        }
        return result

    except json.JSONDecodeError:
        return "Invalid JSON format"
    except ValueError:
        return "Invalid score format"


def main():
    strategy = "sft_data_cot_1"
    agent = "single-expert"
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
    sorted_dataset_list = sorted_dataset_list[1:2]
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
            columns=['index', 'node_address', 'is_phishing', 'label', 'confidence', 'positive_features', 'negative_features', 'conclusion', 'node_detail'])
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
                    response_json = explain2analysis(response)
                    print("output_score success !!!")
                    result = process_score_json(response_json)
                    if isinstance(result, dict):
                        positive_features = result["positive_features"]
                        negative_features = result["negative_features"]    
                        conclusion = result["conclusion"]                          
                        label = result["label"]
                        confidence = result["confidence_score"]
                    print(f"positive_features: {positive_features}")
                    print(f"negative_features: {negative_features}")
                    print(f"conclusion: {conclusion}")                     
                    print(f"label: {label}")
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
            print(f"Label: {label}, Confidence: {confidence}, Predict time： {execution_time_A} s")

            # 将结果追加到 DataFrame
            new_row = pd.DataFrame({
                'index': [i + 1],
                'node_address': [node],
                'is_phishing': [is_phishing],
                'label': [label],
                'confidence': [confidence],
                'positive_features': [positive_features],
                'negative_features': [negative_features],
                'conclusion': [conclusion],
                'node_detail': [train_node_detail]
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

