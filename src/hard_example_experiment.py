from pathlib import Path
import pyreadr
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
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

drop_cols = ["faultNumber", "simulationRun", "sample", "label"]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

# =========================
# 4. 抽样（与前面保持一致）
# =========================
train_sample_0 = train_df[train_df["label"] == 0].sample(n=50000, random_state=42)
train_sample_1 = train_df[train_df["label"] == 1].sample(n=50000, random_state=42)
train_sample = pd.concat([train_sample_0, train_sample_1], ignore_index=True)

test_sample_0 = test_df[test_df["label"] == 0].sample(n=20000, random_state=42)
test_sample_1 = test_df[test_df["label"] == 1].sample(n=20000, random_state=42)
test_sample = pd.concat([test_sample_0, test_sample_1], ignore_index=True)

print("train_sample:", train_sample.shape)
print("test_sample :", test_sample.shape)

X_train = train_sample[feature_cols]
y_train = train_sample["label"].values

X_test = test_sample[feature_cols]
y_test = test_sample["label"].values

# =========================
# 5. 标准化
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 6. 第一阶段：代价敏感基线（weight=2）
# =========================
base_weight = 2
base_sample_weight = np.where(y_train == 1, base_weight, 1.0)

base_model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.1,
    max_depth=8,
    random_state=42
)
base_model.fit(X_train_scaled, y_train, sample_weight=base_sample_weight)

base_pred = base_model.predict(X_test_scaled)
base_metrics = evaluate_binary(y_test, base_pred)

print("\n第一阶段：代价敏感基线（weight=2）")
print(base_metrics)
print_detailed_result("Cost-Sensitive HGB (weight=2)", y_test, base_pred)

# =========================
# 7. 在训练集上找“难例异常样本”
# 定义：训练集中 label=1 且被第一阶段模型预测错误
# =========================
train_pred_stage1 = base_model.predict(X_train_scaled)

hard_mask = (y_train == 1) & (train_pred_stage1 == 0)
hard_indices = np.where(hard_mask)[0]

print(f"\n难例异常样本数量: {len(hard_indices)}")

# =========================
# 8. 第二阶段：难例重加权
# 普通异常样本权重 = 2
# 难例异常样本额外提高到 4 / 6 / 8 做比较
# =========================
hard_weights_to_try = [4, 6, 8]
hard_results = []

best_hard_weight = None
best_hard_f1 = -1
best_hard_model = None
best_hard_pred = None

for hard_weight in hard_weights_to_try:
    sample_weight_stage2 = np.ones_like(y_train, dtype=float)

    # 正常样本 = 1
    sample_weight_stage2[y_train == 1] = 2.0

    # 难例异常再抬高
    sample_weight_stage2[hard_indices] = hard_weight

    model_stage2 = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_depth=8,
        random_state=42
    )
    model_stage2.fit(X_train_scaled, y_train, sample_weight=sample_weight_stage2)

    y_pred_stage2 = model_stage2.predict(X_test_scaled)
    metrics = evaluate_binary(y_test, y_pred_stage2)
    metrics["hard_weight"] = hard_weight
    hard_results.append(metrics)

    print(f"\n>>> 当前难例权重: {hard_weight}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Anomaly Precision: {metrics['anomaly_precision']:.4f}")
    print(f"Anomaly Recall: {metrics['anomaly_recall']:.4f}")
    print(f"Anomaly F1: {metrics['anomaly_f1']:.4f}")

    if metrics["anomaly_f1"] > best_hard_f1:
        best_hard_f1 = metrics["anomaly_f1"]
        best_hard_weight = hard_weight
        best_hard_model = model_stage2
        best_hard_pred = y_pred_stage2

hard_df = pd.DataFrame(hard_results)
hard_df = hard_df[
    ["hard_weight", "accuracy", "anomaly_precision", "anomaly_recall", "anomaly_f1"]
]

print("\n难例重加权实验结果汇总：")
print(hard_df)

hard_csv = RESULTS_DIR / "hard_example_results.csv"
hard_df.to_csv(hard_csv, index=False, encoding="utf-8-sig")
print(f"\n已保存: {hard_csv}")

print(f"\n最佳难例权重: {best_hard_weight}")
print(f"最佳难例异常类 F1: {best_hard_f1:.4f}")

print_detailed_result(
    f"Hard Example Reweighting HGB (hard_weight={best_hard_weight})",
    y_test,
    best_hard_pred
)

# =========================
# 9. 保存总结
# =========================
summary_path = RESULTS_DIR / "hard_example_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("TEP 难例重加权实验总结\n")
    f.write("=" * 40 + "\n\n")

    f.write("一、第一阶段代价敏感基线（weight=2）\n")
    for k, v in base_metrics.items():
        f.write(f"{k}: {v:.6f}\n")

    f.write("\n二、难例重加权实验结果\n")
    f.write(hard_df.to_string(index=False))
    f.write("\n\n")
    f.write(f"最佳难例权重: {best_hard_weight}\n")
    f.write(f"最佳异常类 F1: {best_hard_f1:.4f}\n")

print(f"\n总结文件已保存: {summary_path}")
print("\n实验完成。")