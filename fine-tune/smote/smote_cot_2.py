import os
import pathlib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
from scipy.sparse import hstack, vstack

origin_path = '../../dataset/finetune/finetune_cot_2_2'
new_path = '../../dataset/finetune/finetune_cot_2_2_smote'
multiple = 500


# 自定义解析标签函数
def parse_label(output):
    """解析 output，提取标签为 0, 1 或 2"""
    if "mark it as a Non-Phishing node" in output:
        return 0
    elif "mark it as a Phishing node" in output:
        return 1
    elif "the classification of this node is uncertain" in output:
        return 2
    else:
        raise ValueError("Unexpected label format in output.")


# 设置索引范围并遍历数据集
for idx in range(1, 16):
    json_file = f'{origin_path}/MulDiGraph_delay_5_{multiple}_{idx}.json'
    # 读取 JSON 文件
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"File not found: {json_file}")
    # 检查文件是否为空
    if os.path.getsize(json_file) == 0:
        raise ValueError(f"JSON file is empty: {json_file}")
    data = pd.read_json(json_file)

    # 提取数据中的字段
    X_instruction = data['instruction']
    X_input = data['input']
    X_output = data['output']  # output 包含分析内容和标签
    y = X_output.apply(parse_label)  # 根据分析提取标签

    # 使用 TF-IDF 向量化文本（在合并标签为 0, 1, 2 的数据后进行）
    combined_texts = X_instruction + " " + X_input  # 合并文本以构造完整特征空间
    vectorizer = TfidfVectorizer()
    X_combined_tfidf = vectorizer.fit_transform(combined_texts)

    # 将标签为 2 的数据分离出来
    mask_non_2 = y != 2
    mask_2 = y == 2

    X_non_2 = X_combined_tfidf[mask_non_2]
    y_non_2 = y[mask_non_2]

    X_2 = X_combined_tfidf[mask_2]
    y_2 = y[mask_2]

    # 输出原始类别分布
    print("原始类别分布:", Counter(y))

    # 使用 SMOTE 进行过采样
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_non_2, y_non_2)

    # 将标签为 2 的数据添加回过采样后的数据
    X_final = vstack([X_resampled, X_2])
    y_final = np.concatenate([y_resampled, y_2])

    # 创建一个新的 DataFrame 保存过采样后的数据
    resampled_data = pd.DataFrame({'label': y_final})

    # 同步存储过采样后的文本数据
    resampled_instructions = []
    resampled_inputs = []
    resampled_outputs = []

    # 遍历重新生成的标签，并同步生成其他部分
    idx_2 = 0  # 用于跟踪标签为 2 的数据索引
    for label in y_final:
        if label == 0 or label == 1:
            original_idx = np.random.choice(y[y == label].index)
        else:
            # 按顺序添加标签为 2 的样本
            original_idx = y[y == 2].index[idx_2]
            idx_2 += 1  # 更新索引以获取下一个标签为 2 的样本

        # 根据索引同步获取数据
        resampled_instructions.append(X_instruction.iloc[original_idx])
        resampled_inputs.append(X_input.iloc[original_idx])
        resampled_outputs.append(X_output.iloc[original_idx])

    # 填充到 DataFrame
    resampled_data['instruction'] = resampled_instructions
    resampled_data['input'] = resampled_inputs
    resampled_data['output'] = resampled_outputs

    # 确保列的顺序一致
    resampled_data = resampled_data[['instruction', 'input', 'output']]

    # 输出新数据的类别分布
    new_class_distribution = Counter(resampled_data['output'].apply(parse_label))
    total_samples = sum(new_class_distribution.values())
    for label, count in new_class_distribution.items():
        print(f"类别 {label} 数量: {count}")

    # 打印一个过采样的样例
    sample_idx = np.random.randint(0, len(resampled_data))
    print("\n过采样的样例:")
    print(resampled_data.iloc[sample_idx])

    # 保存为 JSON 文件
    pathlib.Path(new_path).mkdir(parents=True, exist_ok=True)
    resampled_data.to_json(f'{new_path}/MulDiGraph_delay_5_{multiple}_{idx}.json', 
                           orient='records', indent=4)
