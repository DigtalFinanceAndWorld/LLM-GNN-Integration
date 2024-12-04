import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 加载数据
file_path = "./nodes_feature.xlsx"  # 替换为文件路径
data = pd.read_excel(file_path)

# 特征和目标变量
X = data.drop(columns=['is_phishing'])  # 特征
y = data['is_phishing']  # 标签

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 逻辑回归建模
model = LogisticRegression(max_iter=10000, class_weight='balanced')
model.fit(X_scaled, y)

# 输出相关性排名
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Absolute_Coefficient': abs(model.coef_[0])  # 绝对值
})

# 按绝对值从高到低排序
coefficients = coefficients.sort_values(by='Absolute_Coefficient', ascending=False)

# 打印排名结果
print("Feature Importance Ranking:")
print(coefficients)

# 如果需要保存为文件，可以使用以下命令
coefficients.to_csv("feature_correlation_ranking_Logistic_Regression.csv", index=False)

