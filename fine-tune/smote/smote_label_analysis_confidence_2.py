import os
import pathlib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
from scipy.sparse import hstack

origin_path = '../../dataset/finetune/finetune_label_analysis_confidence_2_4096'
new_path = '../../dataset/finetune/finetune_label_analysis_confidence_2_4096_smote'
multiple = 500


# 自定义解析标签函数
def parse_label(output):
    """解析 output，提取标签为 0 或 1"""
    if '"label": 0' in output:
        return 0
    else:
        return 1


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

    # 使用 TF-IDF 向量化文本
    vectorizer = TfidfVectorizer()
    X_instruction_tfidf = vectorizer.fit_transform(X_instruction)
    X_input_tfidf = vectorizer.fit_transform(X_input)

    # 合并向量化后的特征
    X = hstack([X_instruction_tfidf, X_input_tfidf])

    # 输出原始类别分布
    print("原始类别分布:", Counter(y))

    # 使用 SMOTE 进行过采样
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # 创建一个新的 DataFrame 保存过采样后的数据
    resampled_data = pd.DataFrame({'label': y_resampled})

    # 同步存储过采样后的文本数据
    resampled_instructions = []
    resampled_inputs = []
    resampled_outputs = []

    # 遍历重新生成的标签，并同步生成其他部分
    for label in y_resampled:
        if label == 0:
            original_idx = np.random.choice(y[y == 0].index)
        else:
            original_idx = np.random.choice(y[y == 1].index)

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

