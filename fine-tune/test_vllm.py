from vllm import LLM, SamplingParams
import math

# 1. 初始化 LLM 实例
llm = LLM(model="/root/autodl-tmp/Meta-Llama-3.1-8B/LLM-Research/Meta-Llama-3___1-8B")

# 2. 定义输入文本和采样参数
input_text = "What is the capital of France?"

# 配置采样参数，例如生成的最大 token 数量
sampling_params = SamplingParams(
    max_tokens=1280,
    temperature=0.0,
    logprobs=5  # 获取 token 的 log probabilities
)

# 3. 执行推理
output = llm.generate(input_text, sampling_params)

# 解析生成的文本
generated_text = output[0].outputs[0].text
print(f"Generated Text: {generated_text}")

# 获取 logprob 数据（键为 token ID，值为 Logprob 对象）
logprob_data = output[0].outputs[0].logprobs[0]
print(f"Logprob Data: {logprob_data}")

# 提取第一个 token 及其 logprob
first_token_id, first_logprob_obj = next(iter(logprob_data.items()))  # 提取第一个条目
token = first_logprob_obj.decoded_token
logprob_value = first_logprob_obj.logprob

# 计算该 token 的概率
probability = math.exp(logprob_value)

# 打印结果
print(f"Token with Highest Probability: {token.strip()}")
print(f"Log Probability: {logprob_value}")
print(f"Probability: {probability:.4f}")
