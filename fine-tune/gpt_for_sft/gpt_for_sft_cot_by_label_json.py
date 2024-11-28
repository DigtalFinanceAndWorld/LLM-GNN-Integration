import random
import re
import json
import time
import os
import pandas as pd
import tiktoken

from describe_single_node_testing_token_cot_json import describe_single_node_test
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
{{
  "task": "You are a blockchain analyst specializing in identifying fraudulent behavior within the Ethereum network. I will provide you with information about a **Target Node** and its true label (**Phishing** or **Non-phishing**). Your task is to analyze the node's behavior and explain why it has been classified with the given label.",
  "task_details": {{
    "methodology": "Use the **Chain-of-Thought (CoT)** methodology to reason through your analysis, combining observations, logical inference, and conclusions.",
    "dimensions": [
      {{
        "dimension": "Node Transaction Activity",
        "steps": [
          {{
            "step": "Observation",
            "description": "Evaluate the node's in-degree and out-degree, focusing on its overall activity."
          }},
          {{
            "step": "Inference",
            "description": "A high number of total external transactions suggests significant activity, which, combined with high centrality, could indicate risky behavior."
          }},
          {{
            "step": "Conclusion",
            "description": "Determine if the node's transaction activity is abnormally high and whether its central role in the network raises concerns."
          }}
        ]
      }},
      {{
        "dimension": "Transaction Amount and Distribution",
        "steps": [
          {{
            "step": "Observation",
            "description": "Examine the frequency and ratio of zero-value and small-value transactions for both incoming and outgoing activities."
          }},
          {{
            "step": "Inference",
            "description": "High proportions of such transactions often signify obfuscation or deceptive intent, a pattern frequently observed in phishing nodes."
          }},
          {{
            "step": "Conclusion",
            "description": "Determine if these patterns indicate deliberate efforts to confuse tracking mechanisms or simulate legitimate activity."
          }}
        ]
      }},
      {{
        "dimension": "Balance Fluctuation",
        "steps": [
          {{
            "step": "Observation",
            "description": "Analyze the node's balance for sharp changes, including significant inflows and rapid outflows."
          }},
          {{
            "step": "Inference",
            "description": "Large inflows followed by immediate and frequent outflows suggest fund distribution to obscure origins."
          }},
          {{
            "step": "Conclusion",
            "description": "Evaluate whether the balance fluctuations are consistent with phishing node behavior."
          }}
        ]
      }},
      {{
        "dimension": "Transaction Frequency and Timing",
        "steps": [
          {{
            "step": "Observation",
            "description": "Review the node's transaction timeline, particularly its activity following large transfers."
          }},
          {{
            "step": "Inference",
            "description": "Frequent bursts of transactions after large inflows may indicate an attempt to distribute funds quickly and evade tracking."
          }},
          {{
            "step": "Conclusion",
            "description": "Assess whether these patterns align with malicious fund movement strategies."
          }}
        ]
      }},
      {{
        "dimension": "Interaction with Other Nodes",
        "steps": [
          {{
            "step": "Observation",
            "description": "Evaluate the node's connections with known phishing nodes, unknown entities, and its overall network interactions."
          }},
          {{
            "step": "Inference",
            "description": "High outgoing interactions with phishing or unknown nodes raise significant red flags, as they are indicative of malicious intent."
          }},
          {{
            "step": "Conclusion",
            "description": "Determine whether the node's connections suggest coordinated phishing activities."
          }}
        ]
      }},
      {{
        "dimension": "Internal Transactions",
        "steps": [
          {{
            "step": "Observation",
            "description": "Investigate the volume and frequency of internal transactions."
          }},
          {{
            "step": "Inference",
            "description": "Frequent low-value internal transactions often indicate self-circulation, a common tactic for obfuscating fund origins."
          }},
          {{
            "step": "Conclusion",
            "description": "Assess whether the internal transaction patterns are consistent with attempts to hide financial flows."
          }}
        ]
      }}
      {{
        "dimension": "Final Classification and Analysis",
        "steps": [
          {{
            "step": "Summary",
            "description": "Summarize the key findings from all dimensions, highlighting the most significant indicators of phishing or non-phishing behavior."
          }},
          {{
            "step": "Decision",
            "description": "Based on the summarized analysis, classify the node as either 'Phishing Node' or 'Non-Phishing Node'."
          }},
          {{
            "step": "Justification",
            "description": "Provide a concise justification for the classification, ensuring it aligns with the evidence presented in the analysis."
          }}
        ]
      }}
    ]
  }},
  "Target Node": "{test_node_detail}",
  "Label": "{label}",
  "output_format": {{
    "response": "Provide a JSON-formatted analysis with no additional output.",
    "format": "json"
  }}
}}
"""
    messages_A = [
        {"role": "system", "content": messages_system_A},
        {"role": "user", "content": messages_input_A},
    ]
    response_A = gpt_chat(messages_A)
    return messages_A, response_A


def main():
    strategy = "sft_cot_data_by_label_json"
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
    sorted_dataset_list = sorted_dataset_list[1:3]
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
        prompts_file_path = "../prompt/SingleExpert.json"
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
            columns=['index', 'node_address', 'is_phishing', 'analysis', 'node_detail'])
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
            train_node_detail, is_phishing, _ = describe_single_node_test(add_histroy_graph, labels_train, node)

            start_time_A = time.time()
            response = ""
            while True:
                try:
                    messages, response = single_expert_model(train_node_detail, message_task_A,
                                                             message_rule, message_attention_A, messages_system_A, is_phishing)
                    print("single_expert_model success !!!")    
                    print(f"response: {response}")
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
            print(f"Predict time： {execution_time_A} s")

            # 将结果追加到 DataFrame
            new_row = pd.DataFrame({
                'index': [i + 1],
                'node_address': [node],
                'is_phishing': [is_phishing],
                'analysis': [response],
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

