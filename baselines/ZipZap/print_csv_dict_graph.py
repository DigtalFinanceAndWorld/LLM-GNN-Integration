import pickle
import pandas as pd


# 读取CSV文件的前几行
def read_csv_head(file_path, n=5):
    # 读取CSV文件的前n行
    df = pd.read_csv(file_path)
    print(df.head().to_string())


def print_pkl(pkl_file):
    with open(pkl_file, "rb") as f:
        G = pickle.load(f)

    # 查看加载后的内容
    print(type(G))  # 查看数据的类型
    print(len(G))  # 查看有多少个账户的交易
    print(list(G.keys())[:5])  # 打印前5个账户的地址

    # 查看某个账户的交易信息（假设要查看第一个账户）
    first_eoa = list(G.keys())[0]
    print(f"Account: {first_eoa}")
    print(G[first_eoa])  # 打印该账户的交易记录


# 使用示例
if __name__ == '__main__':
    # file_path = '/home/a/zmb_workspace/dataset/ZipZap/normal_eoa_transaction_in_slice_1000K.csv'
    # read_csv_head(file_path, 5)  # 读取并打印前5行

    pkl_file = "../data/MulDiGraph/delay_5/1/train_eoa2seq.pkl"
    print_pkl(pkl_file)

