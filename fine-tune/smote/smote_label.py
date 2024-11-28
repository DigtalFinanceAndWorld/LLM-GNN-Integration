import pathlib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np

origin_path = '../../dataset/finetune/finetune_label_analysis_logits_4096'
new_path = '../../dataset/finetune/finetune_label_analysis_logits_4096_smote'
multiple=500

for idx in range(1, 19):
    # 读取JSON文件
    data = pd.read_json(f'{origin_path}/MulDiGraph_delay_5_{multiple}_{idx}.json')

    # 提取特征和标签
    X_instruction = data['instruction']
    X_input = data['input']
    y = data['output'].apply(lambda x: 0 if x == "\n### Output Label\n0\n" else 1)

    # 使用TF-IDF向量化文本
    vectorizer = TfidfVectorizer()
    X_instruction_tfidf = vectorizer.fit_transform(X_instruction)
    X_input_tfidf = vectorizer.fit_transform(X_input)

    # 合并向量化后的特征
    from scipy.sparse import hstack

    X = hstack([X_instruction_tfidf, X_input_tfidf])

    # 输出原始类别分布
    print("原始类别分布:", Counter(y))

    # 使用SMOTE进行过采样
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # 创建一个新的DataFrame保存过采样的数据
    resampled_data = pd.DataFrame({'output': y_resampled})

    # 将原始数据中对应的文本加入到新的DataFrame中
    resampled_instructions = []
    resampled_inputs = []

    for label in y_resampled:
        if label == 0:
            original_idx = np.random.choice(y[y == 0].index)
        else:
            original_idx = np.random.choice(y[y == 1].index)

        resampled_instructions.append(X_instruction.iloc[original_idx])
        resampled_inputs.append(X_input.iloc[original_idx])

    resampled_data['instruction'] = resampled_instructions
    resampled_data['input'] = resampled_inputs
    resampled_data['output'] = resampled_data['output'].apply(lambda x: "\n### Output Label\n0\n" if x == 0 else "\n### Output Label\n1\n")
    resampled_data = resampled_data[['instruction', 'input', 'output']]
    # 输出新数据的类别占比
    new_class_distribution = Counter(resampled_data['output'])
    total_samples = sum(new_class_distribution.values())
    for label, count in new_class_distribution.items():
        print(f"类别 {label} 数量: {count}")
    # 保存为JSON文件
    pathlib.Path(new_path).mkdir(parents=True, exist_ok=True)
    resampled_data.to_json(f'{new_path}/MulDiGraph_delay_5_{multiple}_{idx}.json', orient='records', indent=4)
