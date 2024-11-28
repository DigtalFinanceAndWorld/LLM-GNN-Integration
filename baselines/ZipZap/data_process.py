import math
import os
import pickle
import json
import re

from tqdm import tqdm


def convert_graph_to_eoa2seq(graph_filename, output_filename):
    print("Loading graph structure from:", graph_filename)
    with open(graph_filename, 'rb') as f:
        G = pickle.load(f)
    print("Graph loaded successfully with", G.number_of_nodes(), "nodes and", G.number_of_edges(), "edges.")

    eoa2seq = {}

    print("Processing each account and its transactions...")
    for account in tqdm(G.nodes, desc="Processing accounts"):
        transactions = []  # 存储该账户的所有交易记录

        # 遍历该账户的所有出边，并将其添加到 transactions 列表中
        for _, recipient, key, data in G.out_edges(account, keys=True, data=True):
            timestamp = int(float(data['timestamp']))
            amount = math.ceil(data['amount'])
            transaction = [
                recipient,  # 接收方地址
                data.get('block_number', 0),  # 区块号（如果存在）
                timestamp,  # 时间戳
                amount,  # 交易金额
                'OUT',  # 交易方向（OUT）
                0  # 初始化交易计数为 0，稍后填充
            ]
            transactions.append(transaction)

        # 遍历该账户的所有入边，并将其添加到 transactions 列表中
        for sender, _, key, data in G.in_edges(account, keys=True, data=True):
            timestamp = int(float(data['timestamp']))
            amount = math.ceil(data['amount'])
            transaction = [
                sender,  # 发送方地址
                data.get('block_number', 0),  # 区块号（如果存在）
                timestamp,  # 时间戳
                amount,  # 交易金额
                'IN',  # 交易方向（IN）
                0  # 初始化交易计数为 0，稍后填充
            ]
            transactions.append(transaction)

        # 根据交易对 (source, target) 进行计数
        transaction_count_dict = {}
        for tx in transactions:
            # 根据交易方向（OUT 或 IN）来确定 source 和 target
            if tx[4] == 'OUT':
                source = account  # 当前账户是发送方
                target = tx[0]    # 接收方是 transaction 中的第一个元素
            elif tx[4] == 'IN':
                source = tx[0]    # 发送方是 transaction 中的第一个元素
                target = account  # 当前账户是接收方
            else:
                raise ValueError(f"Unknown transaction direction: {tx[4]}")

            # 使用 source 和 target 作为键进行计数
            direction_key = (source, target)
            if direction_key not in transaction_count_dict:
                transaction_count_dict[direction_key] = 0
            transaction_count_dict[direction_key] += 1

        # 分配伪区块号的逻辑保持不变...
        for tx in transactions:
            if tx[1] == 0:  # 如果区块号缺失
                tx[1] = calculate_block_number(tx[2], transactions)  # 使用 calculate_block_number 计算伪区块号

        # 遍历所有交易记录，并根据 transaction_count_dict 填充交易计数
        for tx in transactions:
            if tx[4] == 'OUT':
                source = account
                target = tx[0]
            else:
                source = tx[0]
                target = account
            direction_key = (source, target)
            tx[5] = transaction_count_dict[direction_key]  # 更新计数（根据新格式，计数在 tx[5]）

        # 根据时间戳对所有交易进行排序
        transactions.sort(key=lambda x: x[2])  # 按时间戳（tx[2]）升序排序

        # 将该账户的所有交易记录添加到字典中
        eoa2seq[account] = transactions

    print(f"Saving eoa2seq data to {output_filename} ...")
    with open(output_filename, 'wb') as f_out:
        pickle.dump(eoa2seq, f_out)
    print(f"Eoa2seq data saved successfully to {output_filename}")


def calculate_block_number(timestamp, transactions):
    """
    根据时间戳计算区块号（伪区块号），可以根据全局时间戳或局部交易时间戳进行推断。
    这里假设每个交易的时间戳都可以映射到区块号，可以根据时间戳的顺序来分配相对的区块号。
    """
    all_timestamps = sorted(set(tx[2] for tx in transactions))  # 所有交易的时间戳去重并排序（tx[2]是时间戳）
    block_number = all_timestamps.index(timestamp) + 1  # 区块号从1开始计数
    return block_number


def extract_positive_addresses_from_json(input_json, output_txt):
    positive_addresses = []

    # 1. 读取 JSON 文件，并提取阳性地址（value 为 1）
    with open(input_json, 'r', encoding='utf8') as jsonfile:
        data = json.load(jsonfile)  # 加载 JSON 文件为字典格式

        # 遍历 JSON 字典的键值对，提取 value 为 1 的地址
        for address, value in data.items():
            if value == 1:  # 如果 value 为 1，则表示是阳性地址
                positive_addresses.append(address)

    # 2. 将提取到的阳性地址写入到 TXT 文件中
    with open(output_txt, 'w', encoding='utf-8') as txt_file:
        for address in positive_addresses:
            txt_file.write(address + '\n')  # 每行写入一个阳性地址

    print(f"Successfully extracted {len(positive_addresses)} positive addresses to {output_txt}.")


def main():
    dataset_list = ["MulDiGraph", "ZipZap"]
    for dataset in dataset_list:
        delay_list = [5]
        for delay in delay_list:
            dataset_dir = f"/home/a/zmb_workspace/product/Phisher_detect/dataset/{dataset}/data/LLM/delay_{delay}"
            dataset_list = []
            if os.path.exists(dataset_dir):
                print(f"Dir exists: {dataset_dir}")
                dataset_list = os.listdir(dataset_dir)
            else:
                print(f"Dir does not exists: {dataset_dir}")

            def natural_sort_key(filename):
                return [int(s) if s.isdigit() else s for s in re.split('([0-9]+)', filename)]

            sorted_dataset_list = sorted(dataset_list, key=natural_sort_key)
            print(f"Dataset_list: {sorted_dataset_list}")

            for index in sorted_dataset_list:
                train_graph_file = f"{dataset_dir}/{index}/train.pkl"
                train_label_file = f"{dataset_dir}/{index}/train_labels.json"
                test_graph_file = f"{dataset_dir}/{index}/test.pkl"
                test_label_file = f"{dataset_dir}/{index}/test_labels.json"

                output_dir = f"../data/{dataset}/delay_{delay}/{index}/"
                os.makedirs(output_dir, exist_ok=True)
                train_output_pkl_file = f"{output_dir}/train_eoa2seq.pkl"
                train_output_txt_file = f"{output_dir}/train_phisher_account.txt"
                test_output_pkl_file = f"{output_dir}/test_eoa2seq.pkl"
                test_output_txt_file = f"{output_dir}/test_phisher_account.txt"

                convert_graph_to_eoa2seq(train_graph_file, train_output_pkl_file)
                extract_positive_addresses_from_json(train_label_file, train_output_txt_file)
                convert_graph_to_eoa2seq(test_graph_file, test_output_pkl_file)
                extract_positive_addresses_from_json(test_label_file, test_output_txt_file)
                print("------------------------------------------------------------------------------------------")


if __name__ == '__main__':
    main()
