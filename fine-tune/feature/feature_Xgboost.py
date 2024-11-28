import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# 加载数据
file_path = "./nodes_feature.xlsx"  # 替换为文件路径
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

# XGBoost 模型训练
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# 模型预测
y_test_pred = xgb_model.predict(X_test_scaled)

# 模型评估
print("Testing Set Performance:")
print(classification_report(y_test, y_test_pred))
print("Accuracy on Test Set:", accuracy_score(y_test, y_test_pred))

# 映射特征编号到原始名称
booster = xgb_model.get_booster()
feature_map = {f"f{i}": feature for i, feature in enumerate(X.columns)}

# 获取重要性并替换名称
importance = booster.get_score(importance_type='gain')
importance_df = pd.DataFrame(
    [(feature_map[k], v) for k, v in importance.items()],
    columns=['Feature', 'Importance']
).sort_values(by='Importance', ascending=False)

# 打印修正后的特征重要性
print("\nFeature Importance Ranking with Original Feature Names:")
print(importance_df)

# 保存到 CSV 文件
output_file = "feature_correlation_ranking_Xgboost.csv"
importance_df.to_csv(output_file, index=False)
print(f"XGBoost feature importance saved to {output_file}")


