# ACT — B601-RS Battery Insertion

本目录用于记录基于 LeRobot ACT（Action Chunking with Transformers）的 B601-RS 电池插槽插入实验。

当前已完成 ACT 真机复现与初步 Rollout 验证。后续将在这里补充独立、可复现的实验记录，包括：

- 数据集 Repo ID、Episode 数量与总帧数
- ACT 模型结构与训练超参数
- 训练命令、Checkpoint 与推理配置
- Temporal Ensembling（TE）设置
- 真机 Rollout 视频、成功率与失败案例
- 与 π0.5、GR00T 的统一任务评估

**注意：** ACT 的数据集版本和训练配置应单独记录，不要与 `pi0.5/` 中的 `test_1102` 数据集统计混为一谈。
