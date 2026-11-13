# π0.5 — B601-RS Battery Insertion

本目录记录使用 LeRobot 对 π0.5 Base 进行电池插槽插入任务微调的实验。

> 当前实验：**Full Fine-Tuning / 双 RTX 4090 / DDP / 50 条真机示教**
>
> 训练目标为 80,000 Steps。训练完成后的最终真机评估结果尚待补充。

## 1. 实验概览

| 项目 | 配置 |
|---|---|
| Task | `Insert the battery into the slot` |
| Robot | Seeed B601-RS |
| Framework | LeRobot |
| Policy | π0.5 |
| Base Model | π0.5 Base |
| Fine-Tuning | Full Fine-Tuning |
| Dataset Repo ID | `seeed_rebot_b601_rs/test_1102` |
| Episodes | 50 |
| Total Frames | 32,666 |
| GPU | 2 × NVIDIA RTX 4090 |
| Distributed Training | Accelerate + PyTorch DDP |
| Mixed Precision | BF16 |
| Per-GPU Batch Size | 4 |
| Global Batch Size | 8 |
| Target Steps | 80,000 |
| Output Directory | `outputs/train/pi05_full_finetune_test1102` |

## 2. 数据集

数据集使用此前采集的 B601-RS 电池插槽任务真机示教：

- **Repo ID：** `seeed_rebot_b601_rs/test_1102`
- **Episodes：** 50
- **Frames：** 32,666
- **Video Backend：** PyAV（`pyav`）

Episode 表示一条独立示教轨迹；Frame 表示轨迹中的时间步。50 Episodes 与 32,666 Frames 是两个不同的数据规模指标。

本次训练命令没有显式指定 `dataset.root`，因此这里仅记录 Repo ID，不将某台机器的缓存目录写成通用数据路径。

## 3. 模型与微调策略

### 3.1 Base Model

本次从本地 π0.5 Base 权重加载：

```text
/home/jing/rebot_lerobot/models/pi05_base
```

这是本次训练机器上的本地路径；在其他机器运行时，需要替换为对应的模型目录。

### 3.2 Full Fine-Tuning

本次不是 LoRA，也不是 Expert-Only：

```text
policy.freeze_vision_encoder = false
policy.train_expert_only = false
```

训练日志中的参数统计：

| 指标 | 数值 |
|---|---:|
| Total Parameters | 3,616,757,520 |
| Learnable Parameters | 3,616,757,520 |
| Trainable Ratio | 100% |

可训练参数量与总参数量相同，验证了本次运行采用全参数更新。视觉语言主干与动作相关模块均未按 Expert-Only 方式冻结。

## 4. 训练环境与分布式配置

### 4.1 GPU / DDP

| 参数 | 设置 | 说明 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1` | 使用两张 GPU |
| `--multi_gpu` | 开启 | Accelerate 多 GPU 启动 |
| `--num_processes` | 2 | 两个训练进程 |
| `--num_machines` | 1 | 单机训练 |
| `--mixed_precision` | `bf16` | BF16 混合精度 |
| `--dynamo_backend` | `no` | 不使用 Torch Dynamo 编译后端 |
| `--num_cpu_threads_per_process` | 2 | 每个进程的 CPU 线程数 |

DDP 的工作方式是每张 GPU 各持有一份完整模型副本，处理不同的数据 Batch，并在反向传播时同步梯度。**两张 GPU 的显存不会被 DDP 自动合并成一块显存。**

### 4.2 Batch Size

```text
Per-GPU Batch Size = 4
Number of GPUs      = 2
Global Batch Size   = 4 × 2 = 8
```

注意：命令中的 `batch_size=4` 是每个进程 / GPU 的 Batch Size；计算全局 Batch Size 时需要乘以进程数。

### 4.3 精度与显存优化

| 参数 | 设置 | 作用 |
|---|---|---|
| `--policy.dtype` | `bfloat16` | 使用 BF16 模型计算精度 |
| `--policy.gradient_checkpointing` | `true` | 以反向传播时重计算部分激活换取显存 |
| `--policy.compile_model` | `false` | 关闭模型编译 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 调整 CUDA 显存分配行为，缓解碎片问题 |

BF16 具有较大的动态范围，适合大模型训练；Gradient Checkpointing 可以降低激活显存占用，但会增加部分重计算开销。

### 4.4 数据加载

| 参数 | 设置 |
|---|---|
| `--num_workers` | 0 |
| `--dataset.video_backend` | `pyav` |

## 5. 首次训练命令

以下命令按本次实验使用的参数整理。运行前需要确认当前环境已安装对应版本的 LeRobot，并将 Base Model 路径替换为本机实际路径。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  --num_cpu_threads_per_process=2 \
  --module lerobot.scripts.lerobot_train \
  --dataset.repo_id=seeed_rebot_b601_rs/test_1102 \
  --dataset.video_backend=pyav \
  --policy.path=/home/jing/rebot_lerobot/models/pi05_base \
  --policy.input_features=null \
  --policy.output_features=null \
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

### 5.1 关键参数注释

| 参数 | 本次设置 | 解释 |
|---|---:|---|
| `steps` | 80,000 | 训练总步数目标 |
| `batch_size` | 4 | 每张 GPU 的 Batch Size |
| `num_workers` | 0 | 不额外创建 DataLoader Worker |
| `save_freq` | 2,000 | 每 2,000 Steps 保存一次 |
| `log_freq` | 10 | 每 10 Steps 记录一次日志 |
| `eval_freq` | 0 | 不执行周期性自动评估 |
| `save_checkpoint` | true | 保存本地训练 Checkpoint |
| `wandb.enable` | true | 启用 W&B 训练指标记录 |
| `policy.push_to_hub` | false | 不自动上传模型到 Hugging Face Hub |

本次命令没有单独覆盖 Optimizer、Learning Rate 或 Scheduler 的配置值，因此不在这里填入未经本次运行配置验证的数值。后续可从本次实际生成的 `train_config.json` 中补全。

## 6. 训练日志观察

约 19K Steps 附近观察到的日志样例：

| 指标 | 日志观察值 |
|---|---:|
| Loss | 约 0.008–0.011 |
| Gradient Norm | 约 0.399–0.528 |
| Learning Rate | 约 9.2e-6 |
| Update Time | 约 2.729–2.735 s |
| Data Time | 约 0.092–0.094 s |

以上是训练日志中的阶段性样例，不代表整个训练过程的平均值或最终结果。

Loss 下降说明模型对训练数据的拟合在改善，但不能直接等同于真机任务成功率。最终需要通过独立 Rollout 评估电池抓取、移动、对准和插入表现。

## 7. Epoch 等效换算

已知：

- 数据集总帧数：32,666
- 全局 Batch Size：8

按样本数量估算：

```text
Steps per equivalent epoch = 32,666 / 8 ≈ 4,083
Equivalent Epochs = Steps × 8 / 32,666
```

| Steps | 等效 Epoch |
|---:|---:|
| 20,000 | 4.90 |
| 30,000 | 7.35 |
| 40,000 | 9.80 |
| 60,000 | 14.69 |
| 80,000 | 19.59 |

这里的 Epoch 是根据累计样本数计算的等效遍历次数，实际采样器是否完整遍历数据集还取决于 LeRobot 的数据加载与采样实现。

## 8. 待补充的实验结果

- [ ] 训练结束时的最终 Step 与耗时
- [ ] 最终 Training Loss / Gradient Norm
- [ ] Optimizer、Learning Rate、Scheduler 的实际配置（从本次 `train_config.json` 核对）
- [ ] 不同 Checkpoint 的真机 Rollout 视频
- [ ] 多次 Rollout 的抓取成功率与插入成功率
- [ ] 对电池初始位置变化的泛化测试
- [ ] 与 ACT、LoRA Rank 16、LoRA Rank 32、Expert-Only 的统一评估对比

## 9. 复现注意事项

1. 本次 Base Model 路径是本机绝对路径，其他机器需要替换。
2. 数据集 Repo ID 需要在当前机器可访问；本仓库不包含原始示教数据。
3. DDP 下全局 Batch Size 等于单卡 Batch Size 乘以进程数。
5. 不要把 Training Loss 直接当作真机成功率。
6. 本仓库只保存实验文档与必要配置，不提交完整模型权重、训练缓存或原始视频。
