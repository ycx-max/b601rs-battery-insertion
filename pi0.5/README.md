# π0.5 Real-Robot Fine-Tuning on Seeed B601-RS

本项目使用 Hugging Face LeRobot 将 Physical Intelligence π0.5 Base 适配到 Seeed B601-RS 单臂电池插槽任务，完成小样本真机数据微调、不同参数更新策略对比、真机部署与 RTC 推理优化。

> **Task:** Insert the battery into slot  
> **Robot:** 1 × Seeed B601-RS Follower + 1 × reBot Arm 102 Leader  
> **Dataset:** seeed_rebot_b601_rs/test_1102  
> **Dataset Size:** 50 Episodes / 32,666 Frames  
> **Training:** LoRA r32 / Expert-Only / Full Fine-Tuning，均完成 80,000 steps  
> **Hardware:** 2 × NVIDIA RTX 4090  
> **Best Real-Robot Result:** Full FT + RTC，21/25（84%）

## 1. Project Overview

项目围绕同一个单臂电池插槽任务，比较不同参数更新范围对 π0.5 适配效果的影响，并在 Full Fine-Tuning checkpoint 上进一步验证 RTC 的推理优化效果。

实验包含三组完成训练的配置：

1. **LoRA r32**：低秩适配，rank 32。
2. **Expert-Only**：训练 Action Expert，冻结视觉语言主干。
3. **Full Fine-Tuning**：开放模型中配置为可训练的参数，进行全量微调。

本项目使用的是 Inference-Time RTC（Guided RTC），作为部署阶段的动作块衔接与在线推理策略，复用 Full Fine-Tuning checkpoint，不需要额外训练一套模型。Training-Time RTC 作为后续技术扩展介绍，尚未纳入本项目真机评测。

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

LoRA（Low-Rank Adaptation）的核心假设是：预训练模型适配下游任务时，有效权重更新可以近似表示为低秩矩阵，而不必直接优化整个权重矩阵。

对于线性层 $`W_0\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}`$，冻结原始权重，并引入两个可训练矩阵：

```math
A\in\mathbb{R}^{r\times d_{\mathrm{in}}},\qquad
B\in\mathbb{R}^{d_{\mathrm{out}}\times r}
```

```math
W_{\mathrm{eff}}=W_0+\frac{\alpha}{r}BA,\qquad
y=W_0x+\frac{\alpha}{r}BAx
```

其中 $`r`$ 为 Rank，$`\alpha`$ 为缩放系数。通常随机初始化 $`A`$、将 $`B`$ 初始化为零，使训练开始时模型与原始预训练模型等价。反向传播更新适配器参数，而非该线性层的原始权重。

原始线性层包含 $`d_{\mathrm{out}}d_{\mathrm{in}}`$ 个权重，LoRA 仅新增 $`r(d_{\mathrm{in}}+d_{\mathrm{out}})`$ 个可训练参数。以 2048 × 2048 的线性层和 Rank = 32 为例，新增参数为 131,072，约为原权重参数量的 3.125%。因此 LoRA 可以减少梯度和优化器状态的显存开销；但冻结的主干仍需参与前向计算，不能将参数减少比例直接等同于显存或计算量减少比例。标准 LoRA 在部署时还可将低秩增量合并到原权重中。

本项目采用 Rank = 32 的 LoRA 配置完成 80,000-step 训练，目标是在小样本条件下以受限的参数更新范围适配新的机器人本体及精细操作任务。

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
| Learnable / Total | 19.17% |

上述 19.17% 是本次运行日志中整个模型的可训练参数占比，并不等于单个 LoRA 矩阵的理论参数比例。尚未逐模块核对 requires_grad 统计，因此不将全部可训练参数归因于低秩适配器。

在 Action Expert 中，q_proj 影响查询向量及注意力匹配，v_proj 影响被注意力聚合的信息；K/O 投影未列入本次记录的 LoRA 目标。State / Action Projection 和 Time MLP 则与本体动作表征及 Flow Matching 时间条件相关。实际命中的模块及其更新方式仍以所用 LeRobot 版本和训练配置为准。

LoRA r32 在本任务上达到 36% 成功率，低于 Expert-Only 与 Full FT。这提示受限的参数适配能力可能不足以覆盖本体差异与精密接触动作，但仅凭当前三组实验不能确定因果关系；进一步可通过 Rank、目标模块和训练预算消融检验这一假设。

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

LoRA 与 Expert-Only 的训练方式不同于上述 Full FT 命令，应分别设置 PEFT Rank 或 Expert-Only 参数更新范围；三组训练实验的最终步数均为 80,000。

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

### 5.2 RTC 原理（Inference-Time RTC）

π0.5 每次生成一个未来动作块（本项目预测长度为 50 步）。同步推理需要在动作块执行结束后等待新预测，可能产生停顿；普通异步推理虽然能够隐藏部分等待时间，但新旧动作块独立采样，直接切换可能出现动作跳变甚至策略反转。

RTC（Real-Time Chunking）将异步推理与动作前缀条件约束结合：机械臂继续执行旧动作队列，模型同时利用最新观测生成下一段动作，并让新预测与旧动作块在时间重叠区域保持一致。

#### 时间对齐与动作承诺

设 $`H`$ 为预测视界，$`s`$ 为两次推理启动之间已经执行的动作步数（原论文中的 execution horizon），$`d`$ 为推理延迟折算的控制步数。若端到端推理延迟为 $`L`$ 秒、控制频率为 $`f`$ Hz，可保守估计：

```math
d=\lceil Lf\rceil
```

新推理开始时，旧动作块还有 $`H-s`$ 个与新动作块重叠的动作。为了让推理期间的动作连续可执行，需要满足 $`d\le H-s`$。

~~~text
新块推理开始时（以 H=50、s=30、d=5 为示意）：

旧块剩余: [已承诺 5 步][其余重叠 15 步]
新动作块: [前缀约束 5 步][软约束 15 步][自由生成 30 步]
推理完成: 跳过新块中已过去的 5 步，接续执行剩余动作
~~~

这里的步数是示意而非实测值。实际切换应依据控制端已经消耗的动作数进行时间对齐，而不是固定跳过预估延迟；旧队列也必须有足够的动作覆盖推理延迟。

#### Flow Matching 中的引导机制

用 $`o`$ 表示当前观测，$`P`$ 表示按新块时间轴对齐、补齐后的旧动作块剩余部分。采用噪声端 $`\tau=0`$、数据端 $`\tau=1`$ 的时间约定，当前去噪动作为 $`X_\tau`$，速度场为 $`v_\theta`$。可以用一步外推估计干净动作：

```math
\widehat X_1=X_\tau+(1-\tau)v_\theta(X_\tau,o,\tau)
```

定义时间步权重掩码 $`m`$：前 $`d`$ 步为 1（已承诺前缀），其他有效重叠步逐渐衰减，非重叠步为 0。原版 RTC 的 EXP soft mask 在远期逐渐放松旧轨迹约束，使策略仍能响应最新观测。

```math
g_\tau=J_{\widehat X_1}(X_\tau)^\top
\big[m\odot(P-\widehat X_1)\big]
```

```math
\frac{\mathrm d X_\tau}{\mathrm d\tau}
=v_\theta(X_\tau,o,\tau)+\lambda_\tau g_\tau
```

其中 $`J`$ 是干净动作估计相对于当前带噪动作的 Jacobian，$`\lambda_\tau`$ 为随 Flow Matching 时间变化且受最大引导权重限制的系数。以上为原论文引导机制的简化表达，具体实现可能采用相反的 Flow Matching 时间方向。

RTC 通过向量-Jacobian 乘积（VJP）在每次去噪时修正速度场，不更新模型参数；这不是对新旧动作做简单平均。旧队列中已经承诺的动作由执行端继续执行，而引导后的新块用于未来的动作续接。

### 5.3 RTC 工程调参与真机部署

项目在 Full Fine-Tuning checkpoint 上启用原版推理时 RTC，重点涉及：

- 当前动作队列的剩余长度与推理触发时机；
- 新旧动作块的时间对齐、队列合并与动作连续性；
- 连续性约束和最新观测响应之间的平衡。

| Parameter | Role |
|---|---|
| queue_threshold | 剩余动作队列达到触发条件时启动下一次推理；需要覆盖端到端推理延迟及适当裕量 |
| inference_delay | 推理期间消耗的控制步数，基于运行时延迟估计 |
| rtc.execution_horizon | LeRobot guided RTC 中用于保持新旧块一致性的步数，与原论文中两次推理启动间隔 $`s`$ 的定义不完全相同 |
| rtc.max_guidance_weight | 限制速度场引导修正的最大权重，平衡连续性与响应性 |
| prefix_attention_schedule | 重叠区域的约束衰减方式，本项目使用 EXP |

RTC 工程调试从 LeRobot 默认/初始配置 `queue_threshold=30`、`execution_horizon=10`、`max_guidance_weight=10` 开始。测量端到端 P95/P99 延迟时发现冷启动污染延迟统计和动作队列错位问题；修复后先将 `queue_threshold` 从 30 调为 20，再将 `max_guidance_weight` 从 10 调为 5，`execution_horizon=10` 与 EXP schedule 保持不变。完整参数依据、50/30 Hz 延迟数据和修复过程见 [RTC_EXPERIMENT.md](./RTC_EXPERIMENT.md)。

### 5.4 Training-Time RTC 与 Inference-Time RTC

原版 RTC 将动作前缀约束放在推理阶段，通过 VJP 引导生成；后续工作 Training-Time Action Conditioning 将该条件生成能力直接学进模型，从而减少部署时的计算开销。

训练时 RTC 在每个训练样本中随机模拟推理延迟 $`d`$，将示教动作块分为干净前缀和带噪后缀。设真实动作块为 $`A`$、噪声为 $`\epsilon\sim\mathcal{N}(0,I)`$，在噪声到数据的时间约定下：

```math
X_i^\tau=
\begin{cases}
A_i, & i<d\\
\tau A_i+(1-\tau)\epsilon_i, & i\ge d
\end{cases}
```

```math
\tau_i=
\begin{cases}
1, & i<d\\
\tau, & i\ge d
\end{cases}
```

前缀使用无噪声的真实动作及对应的干净时间步，后缀保持标准 Flow Matching 加噪方式。仅对后缀计算速度场监督损失（以下为噪声到数据方向的示意）：

```math
\mathcal L_{\mathrm{RTC}}=
\mathbb E\left[
\frac{1}{H-d}\sum_{i=d}^{H-1}
\left\|v_{\theta,i}(X^\tau,o,\boldsymbol{\tau})-(A_i-\epsilon_i)\right\|^2
\right]
```

模型因此学习给定观测与已知动作前缀时的后续动作分布。部署时，使用上一个动作块的已承诺前缀替代训练时的真实前缀，在生成过程中固定该前缀并预测剩余动作，无需额外的 VJP 引导。注意：LeRobot π0.5 的实际时间约定为数据端 0、噪声端 1 时，上述时间标签和速度目标方向需相应反转。

| Aspect | Inference-Time RTC (Guided) | Training-Time RTC (Trained) |
|---|---|---|
| 条件约束发生阶段 | 推理阶段加入引导 | 训练阶段学习前缀条件，部署时使用 |
| Checkpoint | 兼容普通 Flow Matching checkpoint，无需重新训练 | 需要支持动作前缀条件的专门训练或微调 |
| 动作前缀 | 对已承诺动作强引导，可对更远重叠动作施加软约束 | 对已承诺的前 d 步进行硬条件约束 |
| 训练损失 | 原始 Flow Matching 损失 | 前缀不加噪，损失只监督后缀 |
| 推理计算 | 去噪步骤包含额外 VJP | 固定前缀，免去额外引导反向计算 |
| 延迟适配 | 可动态调整约束及引导权重 | 需要合理覆盖预期延迟分布 |
| 本项目状态 | 已完成真机验证，84% | 尚未开展独立训练及真机评测 |

当前 LeRobot 文档中，兼容的 π0.5 版本可以通过 `policy.rtc_training_max_delay` 设置训练时最大模拟延迟，并在部署时使用 `inference.rtc.mode=trained`；实际参数名称和支持情况以安装版本为准。

**后续实验方向：** 在相同数据集与匹配的训练预算下，比较普通 Full FT、Full FT + Guided RTC 与 Training-Time RTC，控制推理步数和控制频率，评估成功率、P95/P99 端到端延迟、动作块边界跳变及任务完成时长。现有 84% 的结果只对应 Guided RTC，不代表 Training-Time RTC 的性能。

## 6. Results

三种 π0.5 微调实验均完成 80,000 steps。当前成功率对比包含三种训练策略及复用 Full FT checkpoint 的 RTC 推理配置，共四组真机评测，每组 25 次。

| Model / Inference | Training Steps | Successes / Trials | Success Rate |
|---|---:|---:|---:|
| LoRA r32 | 80,000 | 9 / 25 | 36% |
| Expert-Only | 80,000 | 13 / 25 | 52% |
| Full Fine-Tuning | 80,000 | 19 / 25 | 76% |
| Full FT + RTC | Reuse Full FT 80k checkpoint | 21 / 25 | **84%** |

详细结果与分析见 [EVALUATION.md](./EVALUATION.md)，训练过程台账见 [EXPERIMENTS.md](./EXPERIMENTS.md)，RTC 的 50/30 Hz 延迟对比、回弹根因和队列对齐修复见 [RTC_EXPERIMENT.md](./RTC_EXPERIMENT.md)。

### 6.1 Experimental Observations

- 在当前任务和数据设置下，LoRA r32、Expert-Only、Full FT 的成功率依次为 36%、52%、76%。
- Full FT + RTC 相较于 Full FT 提高 8 个百分点，对应 25 次测试中多完成 2 次任务。
- 这些结果用于描述当前 B601-RS 电池插槽任务，不代表跨任务泛化结论。

## 7. Reproduction Checklist

- [x] B601-RS 单臂平台与 Leader-Follower 遥操作数据
- [x] 50 条示教数据及左右初始位置分布
- [x] π0.5 数据预处理与任务 Prompt 记录
- [x] LoRA r32：80,000 steps
- [x] Expert-Only：80,000 steps
- [x] Full Fine-Tuning：80,000 steps
- [x] Full FT checkpoint + RTC 真机推理评测
- [x] RTC 50/30 Hz 延迟测量与动作队列对齐调试
- [x] 不同微调策略的真机成功率对比

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339)
- [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964)
- [LeRobot Real-Time Chunking Documentation](https://huggingface.co/docs/lerobot/rtc)

- [LeRobot](https://github.com/huggingface/lerobot)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
- [π0.5 Real-Robot Fine-Tuning: Data, RTC and Evaluation](https://www.cnblogs.com/ZikangZhou/p/23390824/pi05-real-robot-finetuning-data-rtc)
