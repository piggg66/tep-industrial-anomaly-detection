# 阶段研究报告

本阶段围绕工业多元时序异常检测任务开展研究。由于原始密炼机数据暂不可直接使用，项目组采用 TEP 开源工业多变量时序数据进行阶段性实验验证。

当前已完成：
1. 数据集调研与选型
2. TEP 数据读取与预处理
3. 正常/异常二分类实验
4. RandomForest 与 HistGradientBoosting 模型对比

当前实验结果表明，HistGradientBoosting 在当前任务上优于 RandomForest，说明开源工业多元时序数据能够支撑项目阶段性研究任务。