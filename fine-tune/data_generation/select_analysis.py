import re
import os
import pandas as pd


def get_cot_1(analysis_file_path):
    df = pd.read_excel(analysis_file_path)
    total_rows = len(df)
    print(f"总行数: {total_rows}")
    # 条件 1：is_phishing == 0 且 score < 60
    condition1 = (df['is_phishing'] == 0) & (df['score'] < 60)
    count_condition1 = condition1.sum()  # 统计满足条件 1 的行数

    # 条件 2：is_phishing == 1 且 score > 40
    condition2 = (df['is_phishing'] == 1) & (df['score'] > 40)
    count_condition2 = condition2.sum()  # 统计满足条件 2 的行数

    # 打印统计结果
    print(f"(is_phishing == 0 且 score < 60) 的行数: {count_condition1}")
    print(f"(is_phishing == 1 且 score > 40) 的行数: {count_condition2}")

    filtered_df = df[condition1 | condition2]
    gpt_dict = {
        row['node_address']: {
            'positive_features': row['positive_features'],
            'negative_features': row['negative_features'],
            'conclusion': row['conclusion'],
            'score': row['score']
        }
        for _, row in filtered_df.iterrows()
    }

    return gpt_dict
        
        
def get_cot_2(analysis_file_path):
    df = pd.read_excel(analysis_file_path)
    total_rows = len(df)
    print(f"总行数: {total_rows}")
    # 条件 1：is_phishing == 0 且 score < 70
    condition1 = (df['is_phishing'] == 0) & (df['score'] < 70)
    count_condition1 = condition1.sum()  # 统计满足条件 1 的行数

    # 条件 2：is_phishing == 1 且 score >= 70
    condition2 = (df['is_phishing'] == 1) & (df['score'] >= 70)
    count_condition2 = condition2.sum()  # 统计满足条件 2 的行数

    # 打印统计结果
    print(f"(is_phishing == 0 且 score < 70) 的行数: {count_condition1}")
    print(f"(is_phishing == 1 且 score >= 70) 的行数: {count_condition2}")

    filtered_df = df[condition1 | condition2]
    gpt_dict = {
        row['node_address']: {
            'analysis': row['analysis'],
            'score': row['score']
        }
        for _, row in filtered_df.iterrows()
    }

    return gpt_dict


def get_all_analysis(analysis_file_path):
    df = pd.read_excel(analysis_file_path)
    total_rows = len(df)

    gpt_dict = {
        row['node_address']: {
            'analysis': row['analysis'],
            'score': row['score']
        }
        for _, row in df.iterrows()
    }

    return gpt_dict
    
    
def get_cot_3_analysis(analysis_file_path):
    df = pd.read_excel(analysis_file_path)
    total_rows = len(df)

    gpt_dict = {
        row['node_address']: {
            'analysis': row['analysis']   
        }
        for _, row in df.iterrows()
    }

    return gpt_dict
    
    
def get_analysis(analysis_file_path):
    df = pd.read_excel(analysis_file_path)
    total_rows = len(df)
    print(f"总行数: {total_rows}")
    # 条件 1：is_phishing == 0 且 score < 70
    condition1 = (df['is_phishing'] == 0) & (df['score'] < 70)
    count_condition1 = condition1.sum()  # 统计满足条件 1 的行数

    # 条件 2：is_phishing == 1 且 score >= 70
    condition2 = (df['is_phishing'] == 1) & (df['score'] >= 70)
    count_condition2 = condition2.sum()  # 统计满足条件 2 的行数

    # 打印统计结果
    print(f"(is_phishing == 0 且 score < 70) 的行数: {count_condition1}")
    print(f"(is_phishing == 1 且 score >= 70) 的行数: {count_condition2}")

    filtered_df = df[condition1 | condition2]
    gpt_dict = {
        row['node_address']: {
            'analysis': row['analysis'],
            'score': row['score'],
            'confidence': row['confidence']
        }
        for _, row in filtered_df.iterrows()
    }

    return gpt_dict


def get_gpt_response(analysis_file_path):
    df = pd.read_excel(analysis_file_path)
    total_rows = len(df)
    print(f"总行数: {total_rows}")
    # 条件 1：is_phishing == 0 且 score < 70 且 confidence > 70
    condition1 = (df['is_phishing'] == 0) & (df['score'] < 70) & (df['confidence'] >= 70)
    count_condition1 = condition1.sum()

    # 条件 2：is_phishing == 1 且 score >= 70 且 confidence > 70
    condition2 = (df['is_phishing'] == 1) & (df['score'] >= 70) & (df['confidence'] >= 70)
    count_condition2 = condition2.sum()

    # 打印统计结果
    print(f"(is_phishing == 0 且 score < 70 且 confidence > 70) 的行数: {count_condition1}")
    print(f"(is_phishing == 1 且 score >= 70 且 confidence > 70) 的行数: {count_condition2}")

    filtered_df = df[condition1 | condition2]
    gpt_dict = {
        row['node_address']: {
            'analysis': row['analysis'],
            'score': row['score'],
            'confidence': row['confidence']
        }
        for _, row in filtered_df.iterrows()
    }

    return gpt_dict


def get_gpt_response_2(analysis_file_path):
    df = pd.read_excel(analysis_file_path)
    total_rows = len(df)
    print(f"总行数: {total_rows}")
    condition1 = (df['is_phishing'] == 0) & (df['score'] < 70) & (df['confidence'] >= 60)
    count_condition1 = condition1.sum()

    condition2 = (df['is_phishing'] == 1) & (df['score'] >= 70) & (df['confidence'] >= 60)
    count_condition2 = condition2.sum()

    # 打印统计结果
    print(f"(is_phishing == 0 且 score < 70 且 confidence >= 60) 的行数: {count_condition1}")
    print(f"(is_phishing == 1 且 score >= 70 且 confidence >= 60) 的行数: {count_condition2}")


    filtered_df = df[condition1 | condition2]

    gpt_dict = {
        row['node_address']: {
            'analysis': row['analysis'],
            'score': row['score'],
            'confidence': row['confidence']
        }
        for _, row in filtered_df.iterrows()
    }

    return gpt_dict


def get_gpt_response_by_label(analysis_file_path):
    df = pd.read_excel(analysis_file_path)
    total_rows = len(df)
    print(f"总行数: {total_rows}")
    # 条件 1：is_phishing == 0 且 confidence > 70
    condition1 = (df['is_phishing'] == 0) & (df['confidence'] >= 80)
    count_condition1 = condition1.sum()

    # 条件 2：is_phishing == 1 且 confidence > 70
    condition2 = (df['is_phishing'] == 1) & (df['confidence'] >= 80)
    count_condition2 = condition2.sum()

    # 打印统计结果
    print(f"(is_phishing == 0 且 confidence > 70) 的行数: {count_condition1}")
    print(f"(is_phishing == 1 且 confidence > 70) 的行数: {count_condition2}")

    filtered_df = df[condition1 | condition2]
    gpt_dict = {
        row['node_address']: {
            'analysis': row['analysis'],
            'confidence': row['confidence']
        }
        for _, row in filtered_df.iterrows()
    }

    return gpt_dict


if __name__ == '__main__':
    dataset_name = "MulDiGraph"
    multiple_list = ["500"]
    delay = 5

    for multiple in multiple_list:
        dataset_dir = f'../../dataset/clustered_graphs/{dataset_name}/{multiple}/delay_{delay}'
        if os.path.exists(dataset_dir):
            print(f"Dir exists: {dataset_dir}")
            dataset_list = os.listdir(dataset_dir)
        else:
            print(f"Dir does not exists: {dataset_dir}")
            continue


        def natural_sort_key(filename):
            return [int(s) if s.isdigit() else s for s in re.split('([0-9]+)', filename)]


        # 确保 sorted_dataset_list 中的元素是按自然顺序排列
        sorted_dataset_list = sorted(dataset_list, key=natural_sort_key)
        sorted_dataset_list = sorted_dataset_list[1:8]
        print(f"Dataset_list: {sorted_dataset_list}")

        for idx, index in enumerate(sorted_dataset_list):
            analysis_file_path = f"../../prompt/result/gpt-4o-mini/single-expert/{dataset_name}/{multiple}/delay_{delay}/sft_data_by_score/{index}.xlsx"
            # analysis_dict = get_analysis(analysis_file_path)
            # for key, value in list(analysis_dict.items())[:5]:  # 打印前5项示例
            #     print(f"{key}: {value}")

            gpt_response = get_cot_2(analysis_file_path)
            # for key, value in list(gpt_response.items())[:5]:  # 打印前5项示例
            #     print(f"{key}: {value}")
