## 当前结果

### Baseline Results

| Method | Accuracy | Anomaly Recall | Anomaly F1 |
|---|---:|---:|---:|
| RandomForest | 0.7743 | 0.5658 | 0.7149 |
| HistGradientBoosting | 0.8161 | 0.6468 | 0.7786 |

### Cost-Sensitive Results

| Anomaly Weight | Accuracy | Anomaly Recall | Anomaly F1 |
|---:|---:|---:|---:|
| 1 | 0.8161 | 0.6468 | 0.7786 |
| 2 | 0.8114 | 0.6768 | 0.7821 |
| 3 | 0.7818 | 0.7349 | 0.7711 |
| 5 | 0.6064 | 0.9191 | 0.7002 |

结论：适度异常加权（weight=2）能够在较小精度损失下提升异常类 Recall 和 F1。

### Threshold Optimization Results

| Threshold | Accuracy | Anomaly Recall | Anomaly F1 |
|---:|---:|---:|---:|
| 0.50 | 0.8114 | 0.6768 | 0.7821 |
| 0.45 | 0.8011 | 0.6996 | 0.7786 |
| 0.40 | 0.7769 | 0.7408 | 0.7685 |
| 0.35 | 0.7215 | 0.8065 | 0.7433 |

结论：当前设置下，默认阈值 0.50 已取得最优异常类 F1。

### Hard Example Reweighting Results

| Hard Weight | Accuracy | Anomaly Recall | Anomaly F1 |
|---:|---:|---:|---:|
| 4 | 0.7367 | 0.7873 | 0.7494 |
| 6 | 0.5280 | 0.9777 | 0.6744 |
| 8 | 0.5026 | 0.9973 | 0.6672 |

结论：当前定义下的难例重加权未超过代价敏感基线，说明过度强化难例会导致误报明显增加。