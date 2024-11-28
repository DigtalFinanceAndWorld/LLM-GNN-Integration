import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# 加载数据
file_path = "./nodes_feature.xlsx"  # 替换为你的文件路径
data = pd.read_excel(file_path)

# 特征和目标变量
X = data.drop(columns=['is_phishing'])  # 特征
y = data['is_phishing']  # 标签

# 数据分割（训练集和测试集）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 随机森林分类器
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 模型预测
y_train_pred = rf_model.predict(X_train_scaled)
y_test_pred = rf_model.predict(X_test_scaled)

# 模型评估
print("Training Set Performance:")
print(classification_report(y_train, y_train_pred))
print("Testing Set Performance:")
print(classification_report(y_test, y_test_pred))
print("Accuracy on Test Set:", accuracy_score(y_test, y_test_pred))

# 随机森林特征重要性分析
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# 输出特征重要性排名
print("\nFeature Importance Ranking:")
print(feature_importances)

# 保存为 CSV 文件
output_file = "feature_correlation_ranking_Random_Forest.csv"
feature_importances.to_csv(output_file, index=False)
print(f"Random Forest feature importance saved to {output_file}")


