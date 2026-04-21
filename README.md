# TEP Industrial Anomaly Detection

基于 Tennessee Eastman Process (TEP) 开源工业多变量时序数据的异常检测项目。

## 项目简介
本项目面向工业多元时序异常检测任务，基于 TEP 数据集开展正常/异常二分类实验，并进一步探索面向异常优先识别的改进方法。

当前已完成：
- TEP 数据读取与预处理
- 正常/异常二分类标签构造
- RandomForest 基线实验
- HistGradientBoosting 对比实验

## 数据说明
使用的核心数据文件包括：
- TEP_FaultFree_Training.RData
- TEP_FaultFree_Testing.RData
- TEP_Faulty_Training.RData
- TEP_Faulty_Testing.RData

说明：仓库默认不上传原始数据文件，仅保留读取代码与实验流程。

## 当前结果

| Method | Accuracy | Anomaly Recall | Anomaly F1 |
|---|---:|---:|---:|
| RandomForest | 0.7743 | 0.5658 | 0.7149 |
| HistGradientBoosting | 0.8161 | 0.6468 | 0.7786 |

## 项目结构

.
├─ data/
├─ docs/
├─ results/
├─ src/
├─ README.md
├─ requirements.txt
└─ .gitignore

## 后续计划
- 完成代价敏感异常优先识别实验
- 加入阈值优化与难例重加权
- 补充特征重要性分析
- 构建简单演示原型

## 环境安装
```bash
pip install -r requirements.txt


---

## 6. `results/results.md` 放这个

```markdown
# Baseline Results

## Task
Normal vs Fault binary classification on TEP dataset.

## Data
- Train sample size: 100000
- Test sample size: 40000
- Features: 52

## Results

| Method | Accuracy | Anomaly Recall | Anomaly F1 |
|---|---:|---:|---:|
| RandomForest | 0.7743 | 0.5658 | 0.7149 |
| HistGradientBoosting | 0.8161 | 0.6468 | 0.7786 |

## Conclusion
HistGradientBoosting outperforms RandomForest on the sampled TEP anomaly detection task, especially on anomaly recall and anomaly F1.