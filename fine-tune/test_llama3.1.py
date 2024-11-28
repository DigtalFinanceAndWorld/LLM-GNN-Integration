from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 确保使用 LLaMA 3.1 的具体模型（以 Hugging Face 格式加载）
MODEL_NAME = "/root/autodl-tmp/Meta-Llama-3.1-8B/LLM-Research/Meta-Llama-3___1-8B"  # 替换成你实际使用的 LLaMA 3.x 模型

# 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# 推理函数：生成回答
def generate_response(prompt, max_length=1280, temperature=0.2):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # 确保使用 GPU
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=False,
            top_k=50,
            max_time=10
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 测试案例
if __name__ == "__main__":
    prompts = [
        "Please explain the basic principles of quantum mechanics.",
        "Write a poem about autumn.",
        "What is the overfitting problem in machine learning? How can it be solved?"
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        response = generate_response(prompt)
        print(f"Response: {response}\n")
