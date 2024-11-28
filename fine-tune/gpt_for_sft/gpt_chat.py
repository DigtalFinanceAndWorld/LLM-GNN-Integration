import tiktoken
from openai import OpenAI


API_ENDPOINT_ca = "https://api.openai-proxy.org/v1"
API_ENDPOINT_openai = "https://api.openai.com/v1"

model_gpt3 = "gpt-3.5-turbo"
model_gpt4 = "gpt-4"
model_gpt4mini = "gpt-4o-mini"
model_gpt4_32k = "gpt-4-32k"


def count_tokens(text, model_name):
    encoding = tiktoken.encoding_for_model(model_name)
    token_list = encoding.encode(text)
    return len(token_list)


def gpt_chat(API, messages, model=model_gpt4mini, temperature=0, max_tokens=None):
    client = OpenAI(
        base_url=API_ENDPOINT_openai,
        api_key=API,
    )
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    response = chat_completion.choices[0].message.content
    return response


def main():
    messages = [
        {"role": "system", "content": """You are a financial analyst at a financial company with the ability to 
deeply understand and analyze transaction data. I will provide you with all the transaction records of a 
certain network node. Your task is to identify whether this node is a phishing scam node based on the 
provided Ethereum transaction records. Each transaction record includes the following information:[
Transaction time (e.g., 2017-05-17-09-01 indicates the transaction time as 09:01 on May 17, 2017), 
the node number involved in the transaction, transaction direction, transaction amount (in Bitcoin)]. Please 
note: A transaction amount of 0 may indicate that the transaction was declined or canceled."""},
        {"role": "user", "content": """Here are transactions for node X: [2017-05-17 09:01 22 Outbound 12.35736465]
[2017-05-17 12:04 215 Outbound 0.06161094] [2017-05-17 12:10 26 Outbound 5.634368]. Please analyze whether node X
is a phishing scam node. Please output in the following format at the end: node i: score. For example, if node X
receives a score of 100, you should output: node X:100. 
Please score your assessment based on the following criteria:
Fraud Node Probability Score:
0: Normal transaction information; not a fraud node.
1-10: Very low; highly unlikely to be a fraud node.
11-20: Low; very low fraud probability.
21-40: Slightly low; lower fraud probability.
41-60: Average; average fraud probability.
61-80: High; higher fraud probability.
81-90: Very high; likely a fraud node.
91-99: Extremely high; very likely a fraud node.
100: Matches all fraud characteristics; definitely a fraud node.
Note: This is a sample scoring table. Feel free to customize it based on specific needs and node characteristics."""}]
    response = gpt_chat(API=API8, model=model_gpt4mini, messages=messages)
    print(response)


if __name__ == '__main__':
    main()
