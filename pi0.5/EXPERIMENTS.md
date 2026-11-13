# π0.5 Experiment Ledger

记录 π0.5 在 Seeed B601-RS 单臂电池插槽任务上的全部微调实验、训练状态与真机评测结果。

## 1. Final Training Status

**所有 π0.5 微调训练实验均已完成 80,000 steps。**

| Experiment | Update Strategy | Main Variable | Final Steps | Status |
|---|---|---|---:|---|
| LoRA r16 | Low-Rank Adaptation | Rank = 16 | 80,000 | Completed |
| LoRA r32 | Low-Rank Adaptation | Rank = 32 | 80,000 | Completed |
| Expert-Only | Train Action Expert | Action Expert 参数更新 | 80,000 | Completed |
| Full Fine-Tuning | Full Fine-Tuning | 模型中配置为可训练的参数 | 80,000 | Completed |

RTC 不属于独立训练实验。Full FT + RTC 使用已完成 80,000 steps 的 Full FT checkpoint，在推理阶段启用 RTC。

## 2. Dataset

| Item | Value |
|---|---|
| Dataset | seeed_rebot_b601_rs/test_1102 |
| Episodes | 50 |
| Frames | 32,666 |
| Task Label | Insert the battery into slot |
| Task Index | 0 |
| Cameras | Front / Side |
| Resolution | 640 × 480 |
| FPS | 30 |
| Initial Position Split | Left 25 / Right 25 |

示教数据按电池初始摆放区域均分，左侧 25 条、右侧 25 条。每 5 条示教调整一次初始摆放位置。

## 3. Training Environment

| Item | Value |
|---|---|
| Conda Environment | lerobot |
| Framework | LeRobot |
| Robot | 1 × Seeed B601-RS |
| Leader | 1 × reBot Arm 102 |
| GPU | 2 × NVIDIA RTX 4090 |
| Distributed Launcher | Accelerate |
| Distributed Backend | PyTorch DDP |
| Mixed Precision | BF16 |
| Video Backend | PyAV |

## 4. Experiment-Specific Configuration

### 4.1 LoRA r16

| Item | Value |
|---|---|
| Method | LoRA |
| Rank | 16 |
| Final Training Steps | 80,000 |
| Status | Completed |

### 4.2 LoRA r32

| Item | Value |
|---|---|
| Method | LoRA |
| Rank | 32 |
| Final Training Steps | 80,000 |
| Status | Completed |

Recorded parameter statistics:

| Metric | Value |
|---|---:|
| Total Parameters | 3,616,757,520 |
| Learnable Parameters | 693,422,112 |

Recorded LoRA target modules:

- Action Expert Attention: q_proj、v_proj
- State / Action Projection: state_proj、action_in_proj、action_out_proj
- Action Time MLP: action_time_mlp_in、action_time_mlp_out

### 4.3 Expert-Only

| Item | Value |
|---|---|
| Method | Train Action Expert |
| Frozen Component | PaliGemma vision-language backbone |
| Final Training Steps | 80,000 |
| Status | Completed |

### 4.4 Full Fine-Tuning

| Item | Value |
|---|---|
| Method | Full Fine-Tuning |
| freeze_vision_encoder | false |
| train_expert_only | false |
| Final Training Steps | 80,000 |
| Status | Completed |

Recorded Full FT DDP configuration:

| Parameter | Value |
|---|---:|
| GPU | 2 × RTX 4090 |
| num_processes | 2 |
| Mixed Precision | BF16 |
| Per-device Batch Size | 4 |
| Global Batch Size | 8 |
| gradient_checkpointing | true |
| compile_model | false |
| video_backend | pyav |

## 5. Experiment Configuration Summary

| Experiment | Key Overrides | Final Steps |
|---|---|---:|
| LoRA r16 | peft.method_type=LORA；peft.r=16 | 80,000 |
| LoRA r32 | peft.method_type=LORA；peft.r=32 | 80,000 |
| Expert-Only | policy.train_expert_only=true | 80,000 |
| Full Fine-Tuning | policy.freeze_vision_encoder=false；policy.train_expert_only=false | 80,000 |

以上为区分各实验策略的关键配置项，不是完整独立启动命令。完整训练环境、数据集与 Full Fine-Tuning 启动配置见 [README.md](./README.md)。

## 5. Real-Robot Evaluation

每组已报告的模型 / 推理配置均完成 25 次真机评测。

| Model / Inference | Checkpoint / Training | Successes | Trials | Success Rate |
|---|---|---:|---:|---:|
| LoRA r32 | 80,000 steps | 9 | 25 | 36% |
| Expert-Only | 80,000 steps | 13 | 25 | 52% |
| Full Fine-Tuning | 80,000 steps | 19 | 25 | 76% |
| Full FT + RTC | Reuse Full FT 80k checkpoint | 23 | 25 | **92%** |

LoRA r16 完成 80,000 steps，但没有纳入当前展示的 25 次成功率对比；不根据其他 Rank 的结果推测其成功率。

## 6. Result Interpretation

### 6.1 Fine-Tuning Scope

在当前数据集、机器人平台和任务设置下：

- LoRA r32：9/25，36%；
- Expert-Only：13/25，52%；
- Full Fine-Tuning：19/25，76%。

本组实验显示，不同参数更新范围对应不同的真机任务表现。该结果是当前实验条件下的观察，不直接推断为其他任务上的普遍规律。

### 6.2 RTC Inference

Full FT + RTC 使用同一 Full FT checkpoint，在推理阶段加入 RTC：

- Full FT：19/25，76%；
- Full FT + RTC：23/25，92%；
- 差值：+16 个百分点，即 25 次测试中多完成 4 次任务。

RTC 是推理策略，不是第五组独立的 80,000-step 训练。

## 7. Evaluation Record Template

后续如需补充逐次测试明细，可按以下字段归档：

| Field | Description |
|---|---|
| trial_id | 测试编号 |
| model_config | LoRA r32 / Expert-Only / Full FT / Full FT + RTC |
| checkpoint_step | 模型训练步数 |
| initial_position | 电池初始摆放区域及位置 |
| grasp_result | 抓取是否成功 |
| alignment_result | 对准是否成功 |
| insertion_result | 插入是否成功 |
| task_success | 整体任务是否成功 |
| failure_stage | 失败阶段 |
| notes | 其他观察 |

## 8. Final Checklist

- [x] LoRA r16 完成 80,000 steps
- [x] LoRA r32 完成 80,000 steps
- [x] Expert-Only 完成 80,000 steps
- [x] Full Fine-Tuning 完成 80,000 steps
- [x] Full FT checkpoint + RTC 推理评测
- [x] 记录 25 次真机评测成功次数与成功率
- [x] 记录数据集、任务文本、相机和机器人配置
