# π0.5 Experiment Ledger

本文件用于汇总 B601-RS 电池插槽任务上的 π0.5 微调实验。它记录实验设置和已观察到的信息，不将不同训练配置的 Loss 直接当作可比指标。

## 1. Experiment Matrix

| Experiment | Update Strategy | Main Variable | Recorded Status |
|---|---|---|---|
| LoRA r16 | Low-rank adaptation | Rank = 16 | 已开展训练 |
| LoRA r32 | Low-rank adaptation | Rank = 32 | 已开展训练；有阶段性 Loss 与参数量记录 |
| Expert-Only | Train action expert only | Restrict trainable modules | 已开展训练 |
| Full Fine-Tuning | Update all configured trainable parameters | Full model update | 已记录双卡 DDP 配置；目标 80,000 Steps，最终完成状态待补 |

## 2. Known Training Records

### LoRA r32

| Metric | Recorded Value |
|---|---:|
| Step 50 Loss | ~0.650 |
| Step 100 Loss | ~0.612 |
| Later Loss Range | ~0.15–0.25 |
| Total Parameters | 3,616,757,520 |
| Learnable Parameters | 693,422,112 |

这些数据是阶段性日志记录，不代表最终收敛结果。

### Shared Dataset

| Item | Value |
|---|---|
| Dataset | `seeed_rebot_b601_rs/test_1102` |
| Episodes | 50 |
| Frames | 32,666 |
| Dataset task label | `slot battery` |
| Task index | 0 |

### Training Environment

| Item | Value |
|---|---|
| Conda environment | `lerobot` |
| Framework | LeRobot |
| GPU | 2 × RTX 4090（双卡实验） |
| Distributed launcher | Accelerate |
| Mixed precision | BF16 |
| Video backend | PyAV |

## 3. Comparison Protocol

为保证对比有意义，后续各实验应尽量固定：

- 数据集版本和训练 / 验证划分；
- 相机输入、图像预处理与 State 归一化；
- 任务文本及 Prompt 构造；
- Global Batch Size 和累计样本访问量；
- 真机初始状态、控制频率与评估次数。

需要分别记录训练目标、参数更新范围、训练预算和真机结果。不能仅凭 LoRA Rank 或单次训练 Loss 判断哪种方案更好。

## 4. Results To Add

| Experiment | Final Step | Training Time | Final Loss | Rollout Success |
|---|---:|---:|---:|---:|
| LoRA r16 | 待补充 | 待补充 | 待补充 | 待评估 |
| LoRA r32 | 待补充 | 待补充 | 待补充 | 待评估 |
| Expert-Only | 待补充 | 待补充 | 待补充 | 待评估 |
| Full Fine-Tuning | 待补充 | 待补充 | 待补充 | 待评估 |

成功率应记录为成功次数 / 总测试次数，并注明测试场景和失败阶段。
