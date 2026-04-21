from pathlib import Path
import pyreadr
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def load_rdata(filename):
    path = DATA_DIR / filename
    print("读取文件:", path)
    result = pyreadr.read_r(str(path))
    df = list(result.values())[0]
    return df

ff_train = load_rdata("TEP_FaultFree_Training.RData")
ff_test = load_rdata("TEP_FaultFree_Testing.RData")
fy_train = load_rdata("TEP_Faulty_Training.RData")
fy_test = load_rdata("TEP_Faulty_Testing.RData")

print("ff_train:", ff_train.shape)
print("ff_test :", ff_test.shape)
print("fy_train:", fy_train.shape)
print("fy_test :", fy_test.shape)

ff_train["label"] = 0
ff_test["label"] = 0
fy_train["label"] = 1
fy_test["label"] = 1

train_df = pd.concat([ff_train, fy_train], ignore_index=True)
test_df = pd.concat([ff_test, fy_test], ignore_index=True)

print("train_df:", train_df.shape)
print("test_df :", test_df.shape)

drop_cols = ["faultNumber", "simulationRun", "sample", "label"]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

train_sample_0 = train_df[train_df["label"] == 0].sample(n=50000, random_state=42)
train_sample_1 = train_df[train_df["label"] == 1].sample(n=50000, random_state=42)
train_sample = pd.concat([train_sample_0, train_sample_1], ignore_index=True)

test_sample_0 = test_df[test_df["label"] == 0].sample(n=20000, random_state=42)
test_sample_1 = test_df[test_df["label"] == 1].sample(n=20000, random_state=42)
test_sample = pd.concat([test_sample_0, test_sample_1], ignore_index=True)

print("train_sample:", train_sample.shape)
print("test_sample :", test_sample.shape)

X_train = train_sample[feature_cols]
y_train = train_sample["label"]

X_test = test_sample[feature_cols]
y_test = test_sample["label"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RandomForest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=16,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)

print("=== RandomForest ===")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("F1:", f1_score(y_test, rf_pred))
print("混淆矩阵:")
print(confusion_matrix(y_test, rf_pred))
print("分类报告:")
print(classification_report(y_test, rf_pred, digits=4))

# HistGradientBoosting
hgb = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.1,
    max_depth=8,
    random_state=42
)
hgb.fit(X_train_scaled, y_train)
hgb_pred = hgb.predict(X_test_scaled)

print("=== HistGradientBoosting ===")
print("Accuracy:", accuracy_score(y_test, hgb_pred))
print("F1:", f1_score(y_test, hgb_pred))
print("混淆矩阵:")
print(confusion_matrix(y_test, hgb_pred))
print("分类报告:")
print(classification_report(y_test, hgb_pred, digits=4))