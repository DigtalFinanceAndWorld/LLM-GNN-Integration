import re
import json
import time
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
from describe_single_node_testing import describe_single_node_test
from add_history import *
from purest_sample_selection import generate_purest_detail

model_id = "/root/autodl-tmp/Meta-Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3___1-8B-Instruct"

# Configure 8-bit quantization with BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit quantization
)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
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
    response_C = llama_chat(messages_C)
    return response_C


def main():
    agent = "single-expert"
    strategy = "purest"
    dataset_name = "MulDiGraph"
    multiple = "2000"
    delay = 5
    index = "8"

    test_dir = f'/root/data/clustered_graphs/{dataset_name}/{multiple}/delay_{delay}'

    start_date_file_path = f'/root/data/{dataset_name}/start_times.json'
    train_start_date = load_json(start_date_file_path)["train"][index]
    test_start_date = load_json(start_date_file_path)["test"][index]
    graph_all_file_path = f'/root/data/{dataset_name}/{dataset_name}.pkl'
    graph_all = load_graph(graph_all_file_path)
    
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
    
    total_time = 0.0  # 确保 total_time 被初始化
    num_nodes = 20
    for i, node in enumerate(graph_test.nodes):
        if i >= num_nodes:
            break

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
        total_time += execution_time_A
        print(f"Score: {str(score)}, Predict time for node {i + 1}: {execution_time_A:.2f} seconds")

    avg_time = total_time / num_nodes
    print(f"\nAverage inference time for {num_nodes} nodes: {avg_time:.2f} seconds")


if __name__ == '__main__':
    main()
