# π0.5 Experiment Ledger

记录 Seeed B601-RS 单臂电池插槽任务上的 π0.5 微调实验。

## 1. Experiment Matrix

| Experiment | Update Strategy | Main Variable | Status |
|---|---|---|---|
| LoRA r16 | Low-Rank Adaptation | Rank = 16 | 已完成训练 |
| LoRA r32 | Low-Rank Adaptation | Rank = 32 | 已完成训练 |
| Expert-Only | Train Action Expert | Action Expert 参数更新 | 已开展训练 |
| Full Fine-Tuning | Full Model Fine-Tuning | 全模型可训练参数更新 | 已启动，目标 80,000 Steps |

## 2. LoRA r32 Training Record

| Metric | Recorded Value |
|---|---:|
| Step 50 Loss | ~0.650 |
| Step 100 Loss | ~0.612 |
| Later Loss Range | ~0.15–0.25 |
| Total Parameters | 3,616,757,520 |
| Learnable Parameters | 693,422,112 |

## 3. Dataset

| Item | Value |
|---|---|
| Dataset | seeed_rebot_b601_rs/test_1102 |
| Episodes | 50 |
| Frames | 32,666 |
| Task Label | slot battery |
| Task Index | 0 |
| Cameras | Front / Side |
| Resolution | 640 × 480 |
| FPS | 30 |

## 4. Training Environment

| Item | Value |
|---|---|
| Conda Environment | lerobot |
| Framework | LeRobot |
| Robot | 1 × Seeed B601-RS |
| Leader | 1 × reBot Arm 102 |
| GPU | 2 × NVIDIA RTX 4090 |
| Distributed Launcher | Accelerate |
| Mixed Precision | BF16 |
| Video Backend | PyAV |

## 5. Rollout Comparison

| Experiment | Final Step | Training Time | Final Loss | Rollout Success |
|---|---:|---:|---:|---:|
| LoRA r16 | 待补充 | 待补充 | 待补充 | 待评估 |
| LoRA r32 | 待补充 | 待补充 | 待补充 | 待评估 |
| Expert-Only | 待补充 | 待补充 | 待补充 | 待评估 |
| Full Fine-Tuning | 目标 80,000 | 待补充 | 待补充 | 待评估 |

## 6. Evaluation Metrics

每组模型记录以下真机指标：

- 成功次数 / 总测试次数；
- 电池抓取成功率；
- 插槽对准成功率；
- 插入成功率；
- 平均任务完成时间；
- 各阶段失败次数。

各组模型采用相同的任务流程和测试场景进行比较。
