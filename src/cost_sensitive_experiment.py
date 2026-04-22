from pathlib import Path
import pyreadr
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
)


# =========================
# 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# =========================
# 工具函数
# =========================
def load_rdata(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    print(f"读取文件: {path}")
    result = pyreadr.read_r(str(path))
    df = list(result.values())[0]
    return df


def evaluate_binary(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "anomaly_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "anomaly_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "anomaly_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def print_detailed_result(title, y_true, y_pred):
    print(f"\n===== {title} =====")
    print("混淆矩阵:")
    print(confusion_matrix(y_true, y_pred))
    print("分类报告:")
    print(classification_report(y_true, y_pred, digits=4))


# =========================
# 1. 读取数据
# =========================
ff_train = load_rdata("TEP_FaultFree_Training.RData")
ff_test = load_rdata("TEP_FaultFree_Testing.RData")
fy_train = load_rdata("TEP_Faulty_Training.RData")
fy_test = load_rdata("TEP_Faulty_Testing.RData")

print("ff_train:", ff_train.shape)
print("ff_test :", ff_test.shape)
print("fy_train:", fy_train.shape)
print("fy_test :", fy_test.shape)

# =========================
# 2. 构造标签
# =========================
ff_train["label"] = 0
ff_test["label"] = 0
fy_train["label"] = 1
fy_test["label"] = 1

# =========================
# 3. 合并数据
# =========================
train_df = pd.concat([ff_train, fy_train], ignore_index=True)
test_df = pd.concat([ff_test, fy_test], ignore_index=True)

print("train_df:", train_df.shape)
print("test_df :", test_df.shape)

# =========================
# 4. 选择特征
# =========================
drop_cols = ["faultNumber", "simulationRun", "sample", "label"]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

print("特征数:", len(feature_cols))

# =========================
# 5. 抽样（与你基线保持一致）
# =========================
train_sample_0 = train_df[train_df["label"] == 0].sample(n=50000, random_state=42)
train_sample_1 = train_df[train_df["label"] == 1].sample(n=50000, random_state=42)
train_sample = pd.concat([train_sample_0, train_sample_1], ignore_index=True)

test_sample_0 = test_df[test_df["label"] == 0].sample(n=20000, random_state=42)
test_sample_1 = test_df[test_df["label"] == 1].sample(n=20000, random_state=42)
test_sample = pd.concat([test_sample_0, test_sample_1], ignore_index=True)

print("train_sample:", train_sample.shape)
print("test_sample :", test_sample.shape)

# =========================
# 6. 构造训练/测试集
# =========================
X_train = train_sample[feature_cols]
y_train = train_sample["label"].values

X_test = test_sample[feature_cols]
y_test = test_sample["label"].values

# =========================
# 7. 标准化
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 8. 代价敏感实验
# =========================
weights_to_try = [1, 2, 3, 5]
cost_sensitive_results = []

print("\n=========================")
print("开始代价敏感实验")
print("=========================")

best_weight = None
best_f1 = -1
best_model = None

for anomaly_weight in weights_to_try:
    print(f"\n>>> 当前异常权重: {anomaly_weight}")

    sample_weight = np.where(y_train == 1, anomaly_weight, 1.0)

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_depth=8,
        random_state=42
    )

    clf.fit(X_train_scaled, y_train, sample_weight=sample_weight)
    y_pred = clf.predict(X_test_scaled)

    metrics = evaluate_binary(y_test, y_pred)
    metrics["anomaly_weight"] = anomaly_weight
    cost_sensitive_results.append(metrics)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Anomaly Precision: {metrics['anomaly_precision']:.4f}")
    print(f"Anomaly Recall: {metrics['anomaly_recall']:.4f}")
    print(f"Anomaly F1: {metrics['anomaly_f1']:.4f}")

    if metrics["anomaly_f1"] > best_f1:
        best_f1 = metrics["anomaly_f1"]
        best_weight = anomaly_weight
        best_model = clf

cost_sensitive_df = pd.DataFrame(cost_sensitive_results)
cost_sensitive_df = cost_sensitive_df[
    ["anomaly_weight", "accuracy", "anomaly_precision", "anomaly_recall", "anomaly_f1"]
]

print("\n代价敏感实验结果汇总：")
print(cost_sensitive_df)

cost_sensitive_csv = RESULTS_DIR / "cost_sensitive_results.csv"
cost_sensitive_df.to_csv(cost_sensitive_csv, index=False, encoding="utf-8-sig")
print(f"\n已保存: {cost_sensitive_csv}")

print(f"\n最佳异常权重: {best_weight}")
print(f"最佳异常类 F1: {best_f1:.4f}")

# 用最佳权重模型打印详细结果
best_pred = best_model.predict(X_test_scaled)
print_detailed_result(f"Best Cost-Sensitive HGB (weight={best_weight})", y_test, best_pred)

# =========================
# 9. 阈值优化实验
# =========================
print("\n=========================")
print("开始阈值优化实验")
print("=========================")

y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

thresholds = [0.50, 0.45, 0.40, 0.35]
threshold_results = []

best_threshold = None
best_threshold_f1 = -1
best_threshold_pred = None

for th in thresholds:
    y_pred_th = (y_prob >= th).astype(int)
    metrics = evaluate_binary(y_test, y_pred_th)
    metrics["threshold"] = th
    threshold_results.append(metrics)

    print(f"\n>>> 当前阈值: {th}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Anomaly Precision: {metrics['anomaly_precision']:.4f}")
    print(f"Anomaly Recall: {metrics['anomaly_recall']:.4f}")
    print(f"Anomaly F1: {metrics['anomaly_f1']:.4f}")

    if metrics["anomaly_f1"] > best_threshold_f1:
        best_threshold_f1 = metrics["anomaly_f1"]
        best_threshold = th
        best_threshold_pred = y_pred_th

threshold_df = pd.DataFrame(threshold_results)
threshold_df = threshold_df[
    ["threshold", "accuracy", "anomaly_precision", "anomaly_recall", "anomaly_f1"]
]

print("\n阈值优化实验结果汇总：")
print(threshold_df)

threshold_csv = RESULTS_DIR / "threshold_optimization_results.csv"
threshold_df.to_csv(threshold_csv, index=False, encoding="utf-8-sig")
print(f"\n已保存: {threshold_csv}")

print(f"\n最佳阈值: {best_threshold}")
print(f"最佳阈值下异常类 F1: {best_threshold_f1:.4f}")

print_detailed_result(f"Threshold Optimization (threshold={best_threshold})", y_test, best_threshold_pred)

# =========================
# 10. 保存最终总结
# =========================
summary_path = RESULTS_DIR / "cost_sensitive_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("TEP 异常优先识别实验总结\n")
    f.write("=" * 40 + "\n\n")
    f.write("一、代价敏感实验结果\n")
    f.write(cost_sensitive_df.to_string(index=False))
    f.write("\n\n")
    f.write(f"最佳异常权重: {best_weight}\n")
    f.write(f"最佳异常类 F1: {best_f1:.4f}\n\n")
    f.write("二、阈值优化实验结果\n")
    f.write(threshold_df.to_string(index=False))
    f.write("\n\n")
    f.write(f"最佳阈值: {best_threshold}\n")
    f.write(f"最佳阈值下异常类 F1: {best_threshold_f1:.4f}\n")

print(f"\n总结文件已保存: {summary_path}")
print("\n实验完成。")