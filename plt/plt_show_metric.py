import pandas as pd
from matplotlib import pyplot as plt


def plt_draw(result_dir):
    csv_file_path = f"{result_dir}/result.csv"
    plt_file_path = f"{result_dir}/result.png"
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Convert columns to numeric values where applicable
    df['Precision'] = pd.to_numeric(df['Precision'], errors='coerce')
    df['Recall'] = pd.to_numeric(df['Recall'], errors='coerce')
    df['F1'] = pd.to_numeric(df['F1'], errors='coerce')
    df['TPR@10%FPR'] = pd.to_numeric(df['TPR@10%FPR'], errors='coerce')
    df['AUC'] = pd.to_numeric(df['AUC'], errors='coerce')

    # Filter out any rows that are not numeric experiments (remove summary statistics)
    df_filtered = df[df['Data_num'].apply(lambda x: str(x).isdigit())]

    # Plot Precision, Recall, F1, TPR@10%FPR, AUC over Data_num
    plt.figure(figsize=(12, 8))

    # Plot Precision, Recall, and F1
    plt.plot(df_filtered['Data_num'], df_filtered['Precision'], marker='o', label='Precision')
    plt.plot(df_filtered['Data_num'], df_filtered['Recall'], marker='o', label='Recall')
    plt.plot(df_filtered['Data_num'], df_filtered['F1'], marker='o', label='F1')
    plt.plot(df_filtered['Data_num'], df_filtered['TPR@10%FPR'], marker='o', label='TPR@10%FPR')
    plt.plot(df_filtered['Data_num'], df_filtered['AUC'], marker='o', label='AUC')

    # Adding labels and title
    plt.xlabel('Data Number')
    plt.ylabel('Scores')
    plt.title('Precision, Recall, F1, TPR@10%FPR, and AUC over Data_num')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig(plt_file_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plt png file: {plt_file_path}")


def main():
    result_dir = "../result/ZipZap/deepwalk/delay_30"
    plt_draw(result_dir)


if __name__ == '__main__':
    main()