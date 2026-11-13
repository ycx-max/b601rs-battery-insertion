# π0.5 Real-Robot Evaluation

Seeed B601-RS 单臂电池插槽任务的 π0.5 微调策略与 RTC 推理评估。

## 1. Evaluation Setup

| Item | Configuration |
|---|---|
| Robot | 1 × Seeed B601-RS |
| Leader | 1 × reBot Arm 102 |
| Task | Insert the battery into slot |
| Dataset | seeed_rebot_b601_rs/test_1102 |
| Demonstrations | 50 Episodes |
| Initial Position Split | Left 25 / Right 25 |
| Cameras | Front / Side |
| Resolution | 640 × 480 |
| Camera FPS | 30 |
| Trials per Reported Configuration | 25 |

本页汇总已经完成的 25 次真机评测。成功次数与成功率一一对应。

## 2. Main Results

| Model / Inference Configuration | Training Steps | Successes | Trials | Success Rate |
|---|---:|---:|---:|---:|
| π0.5 LoRA r32 | 80,000 | 9 | 25 | 36% |
| π0.5 Expert-Only | 80,000 | 13 | 25 | 52% |
| π0.5 Full Fine-Tuning | 80,000 | 19 | 25 | 76% |
| π0.5 Full FT + RTC | Reuse Full FT 80k checkpoint | 23 | 25 | **92%** |

LoRA r16 完成 80,000 steps，但没有纳入本页展示的成功率对比。

## 3. Comparison

### 3.1 Fine-Tuning Strategy

在当前真机任务中，三种已评测微调策略的结果为：

- LoRA r32：36%（9/25）
- Expert-Only：52%（13/25）
- Full Fine-Tuning：76%（19/25）

Full Fine-Tuning 相较 LoRA r32 高 40 个百分点，相较 Expert-Only 高 24 个百分点。

### 3.2 RTC Inference

RTC 在 Full Fine-Tuning checkpoint 上作为推理策略启用：

| Configuration | Success Rate |
|---|---:|
| Full Fine-Tuning | 76% |
| Full Fine-Tuning + RTC | **92%** |
| Difference | **+16 percentage points** |

成功次数从 19/25 增加到 23/25，即在这组评测中多完成 4 次任务。

RTC 通过在线处理新旧动作块之间的衔接，改善动作执行阶段的连续性。该对比反映的是当前任务与评测设置下的结果。

## 4. Training Completion

所有 π0.5 微调训练均已完成 80,000 steps：

| Training Run | Final Steps | Status |
|---|---:|---|
| LoRA r16 | 80,000 | Completed |
| LoRA r32 | 80,000 | Completed |
| Expert-Only | 80,000 | Completed |
| Full Fine-Tuning | 80,000 | Completed |

Full FT + RTC 复用 Full FT checkpoint，不单独训练模型。

## 5. Interpretation and Scope

- 所有成功率均针对同一类 B601-RS 单臂电池插槽任务。
- 每个已报告模型 / 推理配置完成 25 次真机评测。
- 当前结果用于比较本项目中的微调与推理配置。
- 25 次测试是有限样本，结果不应直接外推为跨任务、跨机器人或跨环境的泛化性能。
- 当前汇总未提供逐次试验明细、置信区间或显著性检验，因此不对统计显著性作额外判断。

## Related Documents

- [π0.5 Technical & Reproduction Notes](./README.md)
- [π0.5 Experiment Ledger](./EXPERIMENTS.md)
- [Project Overview](../README.md)
- [Real-Robot Demo](../assets/real-robot-demo.mp4)
