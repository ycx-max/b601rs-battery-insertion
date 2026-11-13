# π0.5 Real-Robot Fine-Tuning on Seeed B601-RS

本项目使用 Hugging Face LeRobot 将 Physical Intelligence π0.5 Base 适配到 Seeed B601-RS 单臂电池插槽任务，完成小样本真机数据微调、不同参数更新策略对比、真机部署与 RTC 推理优化。

> **Task:** Insert the battery into slot  
> **Robot:** 1 × Seeed B601-RS Follower + 1 × reBot Arm 102 Leader  
> **Dataset:** seeed_rebot_b601_rs/test_1102  
> **Dataset Size:** 50 Episodes / 32,666 Frames  
> **Training:** LoRA r32 / Expert-Only / Full Fine-Tuning，均完成 80,000 steps  
> **Hardware:** 2 × NVIDIA RTX 4090  
> **Best Real-Robot Result:** Full FT + RTC，23/25（92%）

## 1. Project Overview

项目围绕同一个单臂电池插槽任务，比较不同参数更新范围对 π0.5 适配效果的影响，并在 Full Fine-Tuning checkpoint 上进一步验证 RTC 的推理优化效果。

实验包含四组完成训练的配置：

2. **LoRA r32**：低秩适配，rank 32。
3. **Expert-Only**：训练 Action Expert，冻结视觉语言主干。
4. **Full Fine-Tuning**：开放模型中配置为可训练的参数，进行全量微调。

RTC 是部署阶段的动作块衔接与在线推理策略，复用 Full Fine-Tuning checkpoint，不需要额外训练一套模型。

## 2. Robot and Dataset

### 2.1 Hardware

| Item | Configuration |
|---|---|
| Follower | 1 × Seeed B601-RS |
| Leader | 1 × reBot Arm 102 |
| Framework | LeRobot |
| Policy | π0.5 |
| Base Model | π0.5 Base |
| Camera Views | Front / Side |
| Image Resolution | 640 × 480 |
| Camera FPS | 30 FPS |
| GPU | 2 × NVIDIA RTX 4090 |
| Distributed Training | Accelerate + PyTorch DDP |
| Mixed Precision | BF16 |

### 2.2 Dataset Summary

| Item | Value |
|---|---|
| Dataset Repo ID | seeed_rebot_b601_rs/test_1102 |
| Episodes | 50 |
| Frames | 32,666 |
| Task Text | Insert the battery into slot |
| Task Index | 0 |
| Camera Views | Front / Side |
| Image Resolution | 640 × 480 |
| Camera FPS | 30 FPS |
| Video Backend | PyAV |

50 条真机示教按电池初始摆放区域均分：左侧 25 条、右侧 25 条。每采集 5 条示教调整一次初始摆放位置，以覆盖同一侧的不同相对位置。

### 2.3 Data Representation

LeRobot 数据按时间步关联多视角图像、机器人状态、动作和任务索引。

| Data | Field | Meaning |
|---|---|---|
| Camera Images | observation.images.<camera_key> | Front / Side 当前时刻图像 |
| Robot State | observation.state | 当前机器人状态向量 |
| Action | action | 示教动作，用于动作监督 |
| Task | task_index | 任务文本索引，0 对应 Insert the battery into slot |
| Episode / Frame | episode_index、frame_index、timestamp | 轨迹归属、帧序号与时间戳 |

### 2.4 π0.5 Prompt Construction

数据预处理代码位置：LeRobot 的 pi05 processor。

当前记录的处理流程：

1. 读取当前时间步的机器人 State。
2. 将 State 补齐到模型要求的 max_state_dim。
3. 使用 Normalizer Processor 将 State 归一化到 [-1, 1]。
4. 将归一化 State 离散化为 256 个区间。
5. 将任务文本与离散化 State 组合成 Prompt。
6. 将当前视觉观测与 Prompt 输入模型，以未来连续动作序列作为监督目标。

Prompt 模板：

~~~python
full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
~~~

本项目的任务 Prompt：

~~~text
Task: Insert the battery into slot, State: <discretized state values>;
Action:
~~~

训练样本可抽象为：

- 图像观测：当前 Front / Side 视觉输入。
- 机器人状态：当前时间步 State。
- 语言条件：Insert the battery into slot。
- 动作监督：未来动作序列。

## 3. Fine-Tuning Strategies

### 3.1 Experiment Matrix

所有 π0.5 训练实验均已完成 80,000 steps。

| Experiment | Update Strategy | Rank | Final Steps | Training Status |
|---|---|---:|---:|---|
| LoRA r32 | Low-Rank Adaptation | 32 | 80,000 | Completed |
| Expert-Only | Action Expert fine-tuning | — | 80,000 | Completed |
| Full Fine-Tuning | Full model fine-tuning | — | 80,000 | Completed |

### 3.2 LoRA

LoRA 冻结原始权重，通过低秩矩阵表达权重增量：

W' = W + (alpha / r) × B × A

其中 r 为低秩矩阵的 Rank。本项目采用 Rank = 32 的 LoRA 配置完成 80,000-step 训练。

当前记录的 LoRA 目标模块：

| Module Group | Target Modules |
|---|---|
| Action Expert Attention | q_proj、v_proj |
| State / Action Projection | state_proj、action_in_proj、action_out_proj |
| Action Time MLP | action_time_mlp_in、action_time_mlp_out |

LoRA r32 的训练参数统计记录：

| Metric | Value |
|---|---:|
| Total Parameters | 3,616,757,520 |
| Learnable Parameters | 693,422,112 |

### 3.3 Expert-Only

Expert-Only 将训练范围集中在 Action Expert 相关模块，冻结 PaliGemma 视觉语言主干，使动作专家适配当前机器人数据。该组训练完成 80,000 steps。

### 3.4 Full Fine-Tuning

Full Fine-Tuning 开放模型中配置为可训练的参数，训练配置中：

~~~text
policy.freeze_vision_encoder = false
policy.train_expert_only = false
~~~

该组训练完成 80,000 steps，是后续 RTC 真机推理实验使用的模型 checkpoint。

### 3.5 Experiment Configuration Switches

以下为各训练策略的关键配置差异。它们是训练配置覆盖项，应与数据集、Base Model、设备、日志和保存路径等公共参数组合使用。

| Experiment | Key Configuration |
|---|---|
| LoRA r32 | peft.method_type=LORA；peft.r=32；steps=80000 |
| Expert-Only | policy.train_expert_only=true；steps=80000 |
| Full Fine-Tuning | policy.freeze_vision_encoder=false；policy.train_expert_only=false；steps=80000 |

本项目展示 LoRA r32、Expert-Only 与 Full Fine-Tuning 三种微调策略，均已完成 80,000 steps。

## 4. Training Environment

### 4.1 Hardware and Runtime

| Item | Configuration |
|---|---|
| GPU | 2 × NVIDIA RTX 4090 |
| Distributed Launcher | Accelerate |
| Distributed Backend | PyTorch DDP |
| Mixed Precision | BF16 |
| Video Backend | PyAV |
| Framework | LeRobot |
| Environment | Conda: lerobot |

记录的 Full Fine-Tuning DDP 配置：

| Parameter | Value |
|---|---:|
| num_processes | 2 |
| Per-device Batch Size | 4 |
| Global Batch Size | 8 |
| num_workers | 0 |
| gradient_checkpointing | true |
| compile_model | false |
| Training Steps | 80,000 |

### 4.2 Full Fine-Tuning Command

以下为记录的 Full Fine-Tuning 启动配置。将 PI05_BASE_PATH 替换为本地 π0.5 Base 权重路径后运行。

~~~bash
conda activate lerobot

CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --num_cpu_threads_per_process=8 \
  --module lerobot.scripts.lerobot_train \
  --dataset.repo_id=seeed_rebot_b601_rs/test_1102 \
  --dataset.video_backend=pyav \
  --policy.path=PI05_BASE_PATH \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/pi05_full_finetune_test1102 \
  --job_name=pi05_full_finetune_test1102 \
  --batch_size=4 \
  --num_workers=0 \
  --steps=80000 \
  --eval_freq=0 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=2000 \
  --wandb.enable=true
~~~

LoRA 与 Expert-Only 的训练方式不同于上述 Full FT 命令，应分别设置 PEFT Rank 或 Expert-Only 参数更新范围；四组训练实验的最终步数均为 80,000。

### 4.3 Equivalent Dataset Passes

在全局 Batch Size 为 8 的记录配置下，数据集包含 32,666 Frames：

80,000 × 8 / 32,666 ≈ 19.59

即按该 Batch Size 估算，80,000 steps 对应约 19.59 次数据集等效遍历。该数值是按帧数和全局 Batch Size 计算的估算值，不代表独立的 Epoch 配置。

## 5. Real-Robot Deployment and RTC

### 5.1 Inference Pipeline

1. 读取 Front / Side 双视角图像。
2. 读取机器人 State 并执行 π0.5 对应预处理。
3. 构造任务文本与离散化 State Prompt。
4. 使用微调后的 π0.5 checkpoint 预测动作序列。
5. 将动作序列发送至 B601-RS 执行。
6. 观察抓取、对准和插入结果，记录成功或失败。

### 5.2 RTC

Real-Time Chunking（RTC）用于处理机器人执行当前动作段时，后续动作段的在线生成与衔接问题。项目在 Full Fine-Tuning checkpoint 上启用 RTC，重点涉及：

- 当前动作队列的剩余长度；
- 后续动作块的推理触发时机；
- 新旧动作块的时间对齐；
- 动作重叠区域的衔接；
- 在线执行过程中的动作连续性。

**实验口径：** Full FT + RTC 复用 80,000-step Full FT checkpoint。92% 是 RTC 推理配置下的真机评测结果，不代表 RTC 单独训练了一个新模型。

## 6. Results

四种 π0.5 训练实验均完成 80,000 steps。当前成功率对比包含完成 25 次真机评测的三种训练策略，以及 Full FT checkpoint 的 RTC 推理配置。

| Model / Inference | Training Steps | Successes / Trials | Success Rate |
|---|---:|---:|---:|
| LoRA r32 | 80,000 | 9 / 25 | 36% |
| Expert-Only | 80,000 | 13 / 25 | 52% |
| Full Fine-Tuning | 80,000 | 19 / 25 | 76% |
| Full FT + RTC | Reuse Full FT 80k checkpoint | 23 / 25 | **92%** |

详细结果与分析见 [EVALUATION.md](./EVALUATION.md)，训练过程台账见 [EXPERIMENTS.md](./EXPERIMENTS.md)。

### 6.1 Experimental Observations

- 在当前任务和数据设置下，LoRA r32、Expert-Only、Full FT 的成功率依次为 36%、52%、76%。
- Full FT + RTC 相较于 Full FT 提高 16 个百分点，对应 25 次测试中多完成 4 次任务。
- 这些结果用于描述当前 B601-RS 电池插槽任务，不代表跨任务泛化结论。

## 7. Reproduction Checklist

- [x] B601-RS 单臂平台与 Leader-Follower 遥操作数据
- [x] 50 条示教数据及左右初始位置分布
- [x] π0.5 数据预处理与任务 Prompt 记录
- [x] LoRA r32：80,000 steps
- [x] Expert-Only：80,000 steps
- [x] Full Fine-Tuning：80,000 steps
- [x] Full FT checkpoint + RTC 真机推理评测
- [x] 不同微调策略的真机成功率对比

## References

- [LeRobot](https://github.com/huggingface/lerobot)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
- [π0.5 Real-Robot Fine-Tuning: Data, RTC and Evaluation](https://www.cnblogs.com/ZikangZhou/p/23390824/pi05-real-robot-finetuning-data-rtc)
