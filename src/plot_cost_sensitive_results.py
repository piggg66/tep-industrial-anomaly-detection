from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

csv_path = RESULTS_DIR / "cost_sensitive_results.csv"
df = pd.read_csv(csv_path)

print(df)

# 画 Recall 曲线
plt.figure(figsize=(8, 5))
plt.plot(df["anomaly_weight"], df["anomaly_recall"], marker="o")
plt.xlabel("Anomaly Weight")
plt.ylabel("Anomaly Recall")
plt.title("Anomaly Weight vs Recall")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "weight_vs_recall.png", dpi=300)
plt.show()

# 画 F1 曲线
plt.figure(figsize=(8, 5))
plt.plot(df["anomaly_weight"], df["anomaly_f1"], marker="o")
plt.xlabel("Anomaly Weight")
plt.ylabel("Anomaly F1")
plt.title("Anomaly Weight vs F1")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "weight_vs_f1.png", dpi=300)
plt.show()

# 画 Accuracy 曲线
plt.figure(figsize=(8, 5))
plt.plot(df["anomaly_weight"], df["accuracy"], marker="o")
plt.xlabel("Anomaly Weight")
plt.ylabel("Accuracy")
plt.title("Anomaly Weight vs Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "weight_vs_accuracy.png", dpi=300)
plt.show()

print("图像已保存到 results/ 目录下。")