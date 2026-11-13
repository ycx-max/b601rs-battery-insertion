# π0.5 Real-Robot Fine-Tuning on Seeed B601-RS

使用 LeRobot 将 π0.5 适配到 Seeed B601-RS 单臂电池插槽任务。本目录记录数据预处理、模型微调、训练配置和真机评估过程。

> **Robot:** 1 × Seeed B601-RS Follower + 1 × reBot Arm 102 Leader  
> **Dataset:** seeed_rebot_b601_rs/test_1102  
> **Dataset Size:** 50 Episodes / 32,666 Frames  
> **Training Environment:** Conda lerobot  
> **GPU:** 2 × NVIDIA RTX 4090

## 1. Task and Robot Platform

单臂机械臂根据相机图像、机器人 State 和任务文本，完成电池抓取、移动、插槽对准与插入。

| 项目 | 配置 |
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

Base Model 本地路径：

```text
/home/jing/rebot_lerobot/models/pi05_base
```

## 2. Dataset and Preprocessing

### 2.1 Dataset Summary

| 项目 | 记录 |
|---|---|
| Dataset Repo ID | seeed_rebot_b601_rs/test_1102 |
| Episodes | 50 |
| Frames | 32,666 |
| Video Backend | pyav |
| Camera Views | Front / Side |
| Image Resolution | 640 × 480 |
| Camera FPS | 30 FPS |

### 2.2 Task Description

数据集的 meta/tasks.parquet 内容：

| task 文本 | task_index |
|---|---:|
| slot battery | 0 |

### 2.3 π0.5 Prompt Construction

LeRobot 的 π0.5 预处理代码位于：

```text
lerobot/src/lerobot/policies/pi05/processor_pi05.py
```

预处理过程：

1. 读取机器人 State。
2. 将 State 补齐到 max_state_dim。
3. 使用 Normalizer Processor 将 State 归一化到 [-1, 1]。
4. 将归一化 State 离散化为 256 个区间。
5. 将任务文本与离散化 State 拼接成 Prompt。

代码中的 Prompt 模板：

```python
full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
```

当前数据集生成的 Prompt 结构：

```text
Task: slot battery, State: <discretized state values>;
Action:
```

其中 State 数值由当前时间步的机器人状态计算得到。模型输入包含任务文本、离散化 State 和视觉观测，动作预测由连续机器人动作数据监督。

### 2.4 Training Sample

一个训练样本可以表示为：

$$
(\mathcal{I}_t,\ \mathbf{s}_t,\ \ell,\ \mathbf{A}_t)
$$

- $\mathcal{I}_t$：当前时刻的相机图像。
- $\mathbf{s}_t$：机器人 State。
- $\ell$：任务文本。
- $\mathbf{A}_t$：未来动作序列。

## 3. π0.5 Fine-Tuning Strategies

本项目开展了三种参数更新范围的实验：LoRA、Expert-Only 和 Full Fine-Tuning。

### 3.1 LoRA

LoRA 冻结原模型权重，通过低秩矩阵表示权重更新：

$$
\mathbf{W}' = \mathbf{W} + \frac{\alpha}{r}\mathbf{B}\mathbf{A}
$$

其中：

- $\mathbf{W}$：预训练权重。
- $\mathbf{A}$、$\mathbf{B}$：可训练低秩矩阵。
- $r$：LoRA Rank。
- $\alpha$：缩放系数。

当前代码中的 LoRA 目标模块：

| 模块 | 目标层 |
|---|---|
| Action Expert Attention | q_proj、v_proj |
| State / Action Projection | state_proj、action_in_proj、action_out_proj |
| Action Time MLP | action_time_mlp_in、action_time_mlp_out |

实验配置：

| Experiment | Rank | Status |
|---|---:|---|
| LoRA r16 | 16 | 已完成训练 |
| LoRA r32 | 32 | 已完成训练 |

### 3.2 Expert-Only

Expert-Only 将训练集中在 Action Expert 相关模块，冻结 PaliGemma 视觉语言主干，适配机器人动作生成能力。

当前已开展 Expert-Only 训练，训练结果记录在 pi0.5/EXPERIMENTS.md。

### 3.3 Full Fine-Tuning

Full Fine-Tuning 更新模型中配置为可训练的参数。当前训练配置：

```text
policy.freeze_vision_encoder = false
policy.train_expert_only = false
```

该组实验使用双卡 DDP，训练目标设置为 80,000 Steps。

## 4. Training Configuration

### 4.1 DDP Configuration

| 参数 | 配置 | 含义 |
|---|---:|---|
| GPU | 2 × RTX 4090 | 双卡训练 |
| num_processes | 2 | 两个训练进程 |
| mixed_precision | bf16 | BF16 混合精度 |
| batch_size | 4 | 每个进程的 Batch Size |
| Global Batch Size | 8 | 4 × 2 |
| num_workers | 0 | DataLoader Worker 数量 |
| gradient_checkpointing | true | 激活重计算，降低显存占用 |
| compile_model | false | 关闭模型编译 |
| video_backend | pyav | 视频读取后端 |

### 4.2 Full Fine-Tuning Command

```bash
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
  --policy.path=/home/jing/rebot_lerobot/models/pi05_base \
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
```

### 4.3 Epoch Calculation

数据集包含 32,666 Frames，双卡全局 Batch Size 为 8。等效 Epoch 计算公式：

$$
\mathrm{Epochs} = \frac{S \times B}{N}
$$

其中 $S$ 为训练 Steps，$B$ 为 Global Batch Size，$N$ 为数据集 Frame 数量。

每个等效 Epoch 对应的训练步数：

$$
\frac{32,666}{8} \approx 4,083\ \mathrm{Steps}
$$

| Training Steps | Equivalent Epochs |
|---:|---:|
| 20,000 | 4.90 |
| 30,000 | 7.35 |
| 40,000 | 9.80 |
| 60,000 | 14.69 |
| 80,000 | 19.59 |

## 5. Training Results

### 5.1 LoRA r32

| Training Step | Loss |
|---:|---:|
| 50 | 约 0.650 |
| 100 | 约 0.612 |
| 后续训练阶段 | 约 0.15–0.25 |

参数统计：

| Metric | Value |
|---|---:|
| Total Parameters | 3,616,757,520 |
| Learnable Parameters | 693,422,112 |

### 5.2 Training Metrics

训练期间重点观察：

- Loss：动作预测误差。
- Gradient Norm：梯度变化。
- Learning Rate：学习率调度。
- Update Time：单步参数更新耗时。
- Data Loading Time：数据读取耗时。

## 6. Real-Robot Evaluation

π0.5 模型部署到 B601-RS 单臂平台进行电池插槽任务测试。各组模型使用相同的初始位置和任务流程进行对比。

| Evaluation Item | Description |
|---|---|
| Checkpoint | 模型路径与训练 Step |
| Initial State | 电池初始位置与姿态 |
| Grasp | 电池抓取情况 |
| Alignment | 插槽对准情况 |
| Insertion | 电池插入情况 |
| Completion Time | 任务完成时间 |
| Result | 成功 / 失败 |
| Failure Stage | 失败发生阶段 |

各组实验的最终训练步数、耗时、Loss 和真机 Rollout 统计见 [实验记录表](./EXPERIMENTS.md)。

## 7. RTC and Asynchronous Inference

RTC 关注机器人执行当前动作段时并行生成后续动作段，并通过时间对齐和动作衔接处理新旧动作序列。

单臂 B601-RS 的 RTC 部署重点包括：

- 动作队列剩余长度；
- 新一轮推理的触发时机；
- 模型推理耗时；
- 新旧动作段的重叠区域；
- 动作队列衔接与执行连续性。

## 8. References

- [π0.5 Real-Robot Fine-Tuning: Data, RTC and Evaluation](https://www.cnblogs.com/ZikangZhou/p/23390824/pi05-real-robot-finetuning-data-rtc)
- [LeRobot](https://github.com/huggingface/lerobot)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
