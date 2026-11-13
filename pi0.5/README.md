# π0.5 Real-Robot Fine-Tuning on Seeed B601-RS

本目录记录使用 LeRobot 将 π0.5 适配到 Seeed B601-RS 电池插槽任务的过程。文档重点放在真实实验链路：数据如何进入模型、不同微调方式如何设置、训练预算如何计算，以及最终怎样通过真机 Rollout 判断模型是否有效。

> **当前任务：** Battery grasping and slot insertion  
> **数据集：** \`seeed_rebot_b601_rs/test_1102\`  
> **数据规模：** 50 Episodes / 32,666 Frames  
> **训练环境：** Conda \`lerobot\`  
> **硬件：** 2 × NVIDIA RTX 4090（部分实验使用单卡，部分实验使用双卡 DDP）

## 1. Task and Robot Platform

### 1.1 任务定义

机器人需要根据视觉观测和自身状态，完成电池抓取、移动、对准以及插入插槽。任务是否完成，应以电池是否稳定进入插槽为准，不能只用抓取成功或训练 Loss 下降代替完整任务成功率。

### 1.2 实验平台

| 项目 | 配置 |
|---|---|
| Robot | Seeed B601-RS |
| Framework | LeRobot |
| Policy | π0.5 |
| Base Model | π0.5 Base |
| Task | Battery insertion into slot |
| Training Environment | Conda environment: \`lerobot\` |
| GPU | 2 × NVIDIA RTX 4090 |
| Distributed Training | Accelerate + PyTorch DDP（双卡实验） |
| Mixed Precision | BF16 |

本机 Base Model 路径：

\`\`\`text
/home/jing/rebot_lerobot/models/pi05_base
\`\`\`

这是实验机器上的本地路径，不是仓库内的模型文件。其他机器复现时，需要将其替换为自己的模型目录。

## 2. Dataset and Preprocessing

### 2.1 数据集概况

| 项目 | 记录 |
|---|---|
| Dataset Repo ID | \`seeed_rebot_b601_rs/test_1102\` |
| Episodes | 50 |
| Frames | 32,666 |
| Video Backend | \`pyav\` |
| Camera Views | Front / Side |
| Image Resolution | 640 × 480 |
| Camera FPS | 30 FPS |

数据来自 B601-RS 电池插槽任务的真机遥操作示教。Episode 是一条完整轨迹，Frame 是轨迹中的单个时间步；两者描述的是不同的数据规模。

### 2.2 数据集中的真实任务文本

检查数据集文件：

\`\`\`text
~/.cache/huggingface/lerobot/
└── seeed_rebot_b601_rs/
    └── test_1102/
        └── meta/
            └── tasks.parquet
\`\`\`

读取结果为：

| task 文本（index） | task_index |
|---|---:|
| \`slot battery\` | 0 |

因此，在这份数据集中，实际任务标签是 **\`slot battery\`**，而不是直接写成 \`Insert the battery into the slot\`。如果后续修改任务描述，应同步检查训练数据中的任务映射和部署时使用的任务文本，避免训练与推理不一致。

### 2.3 π0.5 的文本提示是怎样构造的

LeRobot 的 π0.5 预处理并不是只把任务文本原样传给模型。根据当前代码 \`lerobot/src/lerobot/policies/pi05/processor_pi05.py\`，处理过程包括：

1. 复制机器人 State。
2. 将 State 补齐到 \`max_state_dim\`。
3. 使用前序 Normalizer Processor 将 State 归一化到 \`[-1, 1]\`。
4. 将归一化后的 State 离散化为 256 个区间。
5. 将任务文本和离散化 State 拼接成完整 Prompt。

代码中的 Prompt 模板为：

\`\`\`python
full_prompt = f"Task: {cleaned_text}, State: {state_str};\\nAction: "
\`\`\`

对当前数据集而言，模型接收到的文本前缀形式为：

\`\`\`text
Task: slot battery, State: <discretized state values>;
Action:
\`\`\`

这里的 \`<discretized state values>\` 代表当前样本对应的离散化状态序列，并不是固定不变的占位文本。

**结论：当前 π0.5 训练流程包含文本任务提示。** 训练日志通常不会逐 Batch 打印完整 Prompt，但预处理器会在样本进入模型前构造它。任务文本是否足够明确、是否需要改成更自然的指令，是值得通过控制变量实验验证的问题；仅凭当前任务标签，不能断言它已经导致成功率下降。

### 2.4 训练样本的数据流

从概念上看，一个训练样本包含：

\[
(\mathcal{I}_t,\ \mathbf{s}_t,\ \ell,\ \mathbf{A}_t)
\]

其中：

- \(\mathcal{I}_t\)：当前时刻的相机图像；
- \(\mathbf{s}_t\)：机器人 State；
- \(\ell\)：任务文本；
- \(\mathbf{A}_t\)：对应的未来动作序列。

模型根据视觉观测、任务提示和状态信息预测动作。State 的离散化文本表示与图像输入共同构成模型条件，而动作监督仍由数据集中的连续机器人动作提供。

## 3. π0.5 Fine-Tuning Strategies

本项目记录了不同参数更新范围的实验。比较这些方法时，需要同时记录可训练参数量、有效 Batch Size、训练步数、数据版本和真机评估条件。

### 3.1 LoRA

LoRA 冻结原模型权重，通过低秩矩阵表示权重更新：

\[
\mathbf{W}'=\mathbf{W}+\frac{\alpha}{r}\mathbf{B}\mathbf{A}
\]

其中 \(r\) 是 LoRA Rank。Rank 越大，可表示的更新空间通常越大，但可训练参数和优化器状态也会随之增加。

当前代码中记录的适配目标包括：

- VLM 侧：\`q_proj\`、\`v_proj\`
- State / Action 相关模块：\`state_proj\`、\`action_in_proj\`、\`action_out_proj\`
- 时间条件 MLP：\`action_time_mlp_in\`、\`action_time_mlp_out\`

LoRA 实验记录：

| 实验 | Rank | 训练方式 | 记录状态 |
|---|---:|---|---|
| LoRA r16 | 16 | LoRA 微调 | 已开展训练 |
| LoRA r32 | 32 | LoRA 微调 | 已开展训练 |

### 3.2 Expert-Only

Expert-Only 的目标是将训练范围集中在动作专家相关模块，而不是对整个 VLM 主干进行全参数更新。它适合用来研究：在保留预训练视觉语言表征的条件下，仅适配动作生成部分是否足以完成当前机器人任务。

当前已启动并记录 Expert-Only 实验。最终效果需要结合统一的真机评估判断，不能仅凭参数更新范围推断优劣。

### 3.3 Full Fine-Tuning

Full Fine-Tuning 更新模型中允许训练的全部参数。当前仓库记录的一组全参数微调配置使用：

\`\`\`text
policy.freeze_vision_encoder = false
policy.train_expert_only = false
\`\`\`

该配置与 LoRA、Expert-Only 的参数更新范围不同。文档中的 80,000 Steps 是当前记录的训练目标配置，不代表已经完成，也不代表它一定是最优训练预算。

## 4. Training Configuration

### 4.1 双卡 DDP 配置

| 参数 | 当前记录值 | 含义 |
|---|---:|---|
| GPU 数量 | 2 | 两张 RTX 4090 |
| \`--num_processes\` | 2 | 两个训练进程 |
| \`--mixed_precision\` | \`bf16\` | BF16 混合精度 |
| \`--batch_size\` | 4 | 每个进程 / GPU 的 Batch Size |
| Global Batch Size | 8 | 4 × 2 |
| \`--num_workers\` | 0 | DataLoader 不额外启动 Worker 进程 |
| \`gradient_checkpointing\` | true | 通过重计算部分激活降低显存占用 |
| \`compile_model\` | false | 关闭模型编译 |
| \`video_backend\` | \`pyav\` | 使用 PyAV 读取视频 |

DDP 是数据并行：每张 GPU 持有模型副本，处理不同数据并同步梯度。两张 GPU 的显存不会自动合并成一块显存。

### 4.2 训练命令

下面是仓库当前记录的双卡 Full Fine-Tuning 配置。运行前请确认本机 LeRobot 版本、数据集路径和 Base Model 路径与实验记录一致。

\`\`\`bash
conda activate lerobot

CUDA_VISIBLE_DEVICES=0,1 \\
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
accelerate launch \\
  --multi_gpu \\
  --num_processes=2 \\
  --num_machines=1 \\
  --mixed_precision=bf16 \\
  --num_cpu_threads_per_process=8 \\
  --module lerobot.scripts.lerobot_train \\
  --dataset.repo_id=seeed_rebot_b601_rs/test_1102 \\
  --dataset.video_backend=pyav \\
  --policy.path=/home/jing/rebot_lerobot/models/pi05_base \\
  --policy.device=cuda \\
  --policy.dtype=bfloat16 \\
  --policy.gradient_checkpointing=true \\
  --policy.compile_model=false \\
  --policy.freeze_vision_encoder=false \\
  --policy.train_expert_only=false \\
  --policy.push_to_hub=false \\
  --output_dir=outputs/train/pi05_full_finetune_test1102 \\
  --job_name=pi05_full_finetune_test1102 \\
  --batch_size=4 \\
  --num_workers=0 \\
  --steps=80000 \\
  --eval_freq=0 \\
  --log_freq=10 \\
  --save_checkpoint=true \\
  --save_freq=2000 \\
  --wandb.enable=true
\`\`\`

注意：

- \`--batch_size=4\` 是单个训练进程的 Batch Size；双卡时全局 Batch Size 为 8。
- \`--policy.push_to_hub=false\` 表示不将模型自动推送到 Hugging Face Hub。
- \`--wandb.enable=true\` 仍会记录 W&B 训练日志；它与是否推送模型权重到 Hub 是两回事。
- 该命令是 Full Fine-Tuning 配置，不可直接当作 LoRA 或 Expert-Only 的启动命令。
- 本次运行的优化器、Scheduler 和学习率应以对应输出目录中的实际 \`train_config.json\` 为准；没有核对的配置不在文档中补写。

### 4.3 Epoch 与训练预算

LeRobot 日志中的 Epoch 可按累计样本访问量估算。若数据集包含 \(N=32,666\) Frames，双卡全局 Batch Size 为 \(B=8\)，训练 \(S\) Steps，则：

\[
\mathrm{Equivalent\ Epochs}=\frac{S\times B}{N}
\]

每个等效 Epoch 对应的 Step 数约为：

\[
\frac{32,666}{8}\approx 4,083
\]

| Training Steps | 等效 Epoch |
|---:|---:|
| 20,000 | 4.90 |
| 30,000 | 7.35 |
| 40,000 | 9.80 |
| 60,000 | 14.69 |
| 80,000 | 19.59 |

这里的 Epoch 是按累计样本访问量计算的等效遍历次数。若数据采样器采用随机抽样或带有重复采样，它不一定意味着每条轨迹都被严格、完整地遍历了相同次数。

## 5. Training Observations

### 5.1 LoRA r32

当前记录的 LoRA r32 训练日志包括：

| 训练阶段 | Loss 观察值 |
|---|---:|
| Step 50 | 约 0.650 |
| Step 100 | 约 0.612 |
| 后续训练阶段 | 约 0.15–0.25 |

参数统计记录：

| 指标 | 数值 |
|---|---:|
| Total Parameters | 3,616,757,520 |
| Learnable Parameters | 693,422,112 |

以上数值对应当前记录的 r32 运行。Loss 数值受训练配置、数据预处理和损失定义影响，不应直接与其他训练策略的 Loss 横向比较。

### 5.2 如何判断训练是否有效

训练 Loss 下降只能说明模型在当前训练目标上的拟合误差减小，不能直接说明机器人更容易完成电池插入。真机任务还受到视觉遮挡、末端定位误差、接触过程、夹爪状态和控制时序影响。

因此，不同微调策略至少需要同时比较：

1. 训练 Loss 与 Gradient Norm；
2. 可训练参数量和实际训练预算；
3. 相同数据版本下的真机 Rollout；
4. 抓取、移动、对准、插入各阶段的失败分布；
5. 多次测试的成功次数 / 总次数，而不只展示单次成功视频。

## 6. Real-Robot Evaluation

π0.5 的最终效果应通过真实 B601-RS 机器人评估。建议固定测试场景、初始电池位置范围、控制频率和评估次数，再对比 LoRA r16、LoRA r32、Expert-Only 与 Full Fine-Tuning。

| 评估项 | 记录内容 |
|---|---|
| Checkpoint | 模型路径与训练 Step |
| Initial State | 电池初始位置与姿态 |
| Grasp | 是否稳定抓取 |
| Alignment | 是否完成插槽对准 |
| Insertion | 是否完成插入 |
| Completion Time | 完成任务所需时间 |
| Result | 成功 / 失败 |
| Failure Stage | 首次出现失败的阶段 |

**当前状态：** 本 README 暂不填写未经统一测试协议验证的 π0.5 真机成功率。完成多次 Rollout 后，再将成功次数、总测试次数、失败阶段和视频链接补充到本节。

## 7. RTC and Asynchronous Inference

RTC（并行化推理与执行的相关方法）关注的是：机器人执行当前动作段时，如何并行生成下一段动作，并在新旧动作段之间处理时间对齐与衔接。

它与单纯增加 Action Chunk 长度不同，也不等同于普通的异步线程。实际部署需要记录：

- 当前动作队列剩余长度；
- 新一轮推理的触发时机；
- 模型推理耗时及其抖动；
- 新动作段到达时已经过去的时间步；
- 新旧动作段重叠区域的处理方式；
- 队列是否出现耗尽，以及动作衔接是否平滑。

**当前状态：** RTC 的 B601-RS 接入与对照评估尚待补充。本节作为后续实验记录框架，不代表已经取得 RTC 成功率提升。

## 8. Reproducibility Checklist

- [x] 记录数据集 Repo ID、Episodes 和 Frames
- [x] 核对 \`tasks.parquet\` 中的任务文本映射
- [x] 核对 π0.5 Prompt 构造代码
- [x] 记录 Conda 环境为 \`lerobot\`
- [x] 记录 LoRA r16、LoRA r32 和 Expert-Only 实验
- [x] 记录双卡 DDP 与 BF16 配置
- [ ] 从每组实验的 \`train_config.json\` 补齐优化器和 Scheduler
- [ ] 汇总各实验最终 Checkpoint 与实际训练耗时
- [ ] 使用统一测试协议完成各策略真机 Rollout
- [ ] 记录各阶段成功率、失败案例和完成时间
- [ ] 完成 RTC 接入后补充时序与对照实验

## 9. References

- [π0.5 Real-Robot Fine-Tuning: Data, RTC and Evaluation](https://www.cnblogs.com/ZikangZhou/p/23390824/pi05-real-robot-finetuning-data-rtc) — 参考其真实机器人实验复盘的组织方式；本文中的硬件、数据和训练记录均以本项目实际实验为准。
- [LeRobot](https://github.com/huggingface/lerobot)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
