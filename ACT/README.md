# ACT — B601-RS Battery Insertion

基于 LeRobot ACT（Action Chunking with Transformers）的单臂电池插槽装配实验。本文档记录实际训练配置、示教数据分布、推理设置、真机评估与迭代结果。

## 1. 项目概述

| 项目 | 配置 |
|---|---|
| Robot | Seeed B601-RS Follower |
| Leader | reBot Arm 102 Leader |
| 操作形式 | 单臂操作 |
| Framework | LeRobot |
| Policy | ACT |
| Task | `Insert the battery into the slot` |
| 训练数据集 | `seeed_rebot_b601_rs/test1012_2` |
| Episodes | 50 |
| Frames | 29,737 |
| 训练设备 | CUDA（W&B run 记录）；GPU 型号以该次运行的系统元数据为准 |
| W&B Project | `lerobot` |
| W&B Run | `act_rebot_test` |
| Run ID | `yidm7mzl` |
| Seed | 1000 |

训练记录与曲线：[查看 W&B Run — act_rebot_test](https://forge.coreweave.com/wandb/3213152090-bit/lerobot/runs/yidm7mzl?nw=nwuser3213152090)

## 2. 单臂任务改造与示教数据设计

### 2.1 与原始双臂任务的区别

原始论文中的任务采用双臂操作，并使用白线辅助遥控器随机摆放；论文没有明确给出电池初始位置的摆放规则。

本项目使用一台 B601-RS Follower 完成单臂复现。由于不再由另一只机械臂协助稳定遥控器，因此**将遥控器固定在工作台上**，主要通过改变电池初始位置引入任务初始状态的变化。


### 2.2 电池初始位置分布

共采集 50 条专家示教轨迹：

| 位置类别 | Episodes | 摆放规则 |
|---|---:|---|
| 与遥控器同侧 | 30 | 电池相对遥控器的位置不同；每 5 条示教更换一次初始位置 |
| 与遥控器异侧 | 20 | 电池放置在遥控器另一侧；每 5 条示教更换一次初始位置 |
| **合计** | **50** | 覆盖不同电池初始位置 |

同侧 30 条示教中，每 5 条切换一次初始位置，对应 6 组相对位置设置。

采集位置统计另记录为左侧 30 Episodes、右侧 20 Episodes。

### 2.3 传感器与数据

- 双视角 RGB 相机：Front / Side。
- 图像分辨率：640 × 480。
- 相机帧率：30 FPS。
- 总数据量：50 Episodes，29,737 Frames。

## 3. ACT 模型配置

以下参数来自此前实际训练记录与 W&B 配置，不是原论文参数表的简单照搬。

| 参数 | 实际配置 | 说明 |
|---|---:|---|
| Vision Backbone | ResNet18 | 使用 ImageNet 预训练视觉骨干 |
| `use_vae` | `true` | 使用 CVAE 潜变量建模示教动作分布 |
| `latent_dim` | 32 | 潜变量维度 |
| `kl_weight` | 10 | KL 正则项权重 |
| `chunk_size` | 100 | 每次预测 100 步动作 |
| `n_obs_steps` | 1 | 输入观测时间步数 |
| `n_action_steps` | 100 | 每次预测后执行 100 步 |
| `n_encoder_layers` | 4 | Transformer Encoder 层数 |
| `n_decoder_layers` | 1 | LeRobot ACT 实际配置 |
| `temporal_ensemble_coeff` | `None` | 未启用 Temporal Ensembling |
| `image_transforms.enable` | `false` | 关闭图像增强 |

### 3.1 Action Chunk 与推理方式

ACT 每次根据当前观测预测一个长度为 100 的动作序列（Action Chunk）。

本次部署采用同步动作块推理，目标控制频率为 30 Hz。按每次执行完整 100 步估算，一个动作块覆盖的时间约为：

```text
100 / 30 ≈ 3.33 seconds
```

本次 `temporal_ensemble_coeff=None`，即未启用时间集成；不能将 ACT 论文中提出的 Temporal Ensembling 机制写成本项目已启用的功能。

## 4. 训练配置

### 4.1 W&B 实际运行信息

| 参数 | 实际值 |
|---|---|
| Run Name | `act_rebot_test` |
| Run ID | `yidm7mzl` |
| Dataset | `seeed_rebot_b601_rs/test1012_2` |
| Policy | `act` |
| Seed | 1000 |
| Device | `cuda` |
| Episodes | 50 |
| Frames | 29,737 |
| Batch Size | 8 |
| Training Steps | 300,000 |
| Checkpoint Save Frequency | 20,000 |

### 4.2 Optimizer

| 参数 | 配置 |
|---|---:|
| Optimizer | AdamW |
| Learning Rate | `1e-5` |
| Weight Decay | `1e-4` |
| Betas | `(0.9, 0.999)` |
| Gradient Clip Norm | 10 |

### 4.3 参数量与等效 Epoch

训练日志记录：

```text
num_episodes = 50
num_frames = 29737
batch_size = 8
steps = 300000
num_learnable_params = 51599239
```

可训练参数约为 51.6M。

按全局 Batch Size 8 粗略估算等效 Epoch：

```text
Equivalent Epochs
= 300000 × 8 / 29737
≈ 80.7
```

这里的 Epoch 是按累计样本访问量折算的等效遍历次数，不表示采集了额外的新数据。ACT 训练样本还包含相互重叠的未来动作窗口。

### 4.4 实际执行的训练命令

历史记录中保存的实际启动命令：

```bash
lerobot-train \
  --dataset.repo_id=seeed_rebot_b601_rs/test1012_2 \
  --policy.type=act \
  --output_dir=outputs/train/act_rebot_test1012_2 \
  --job_name=act_rebot_test \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false \
  --steps=300000
```

说明：

- 上面的命令保留了当时记录的实际 CLI 参数。
- `batch_size=8`、`seed=1000`、`save_freq=20000` 以及 Optimizer 参数来自该次运行记录 / W&B 配置；它们没有全部显式写在这条 CLI 命令中。
- 未经历史配置核实的训练耗时、具体 GPU SKU 和其他默认参数不在此处猜测。

## 5. 训练曲线观察

W&B 中记录了 `train/loss`、`train/l1_loss`、`train/lr`、`train/steps` 等训练指标。

此前保存的曲线观察：

- L1 Loss 从约 6–7 逐渐下降至约 0.02–0.03。
- KL Loss 接近 0，提示需要关注 CVAE 后验是否出现坍缩；仅凭 KL 数值不能单独判定策略效果。
- 训练 Loss 下降说明模型对训练示教的拟合改善，但不能替代独立真机评估。

完整曲线见 [W&B Run](https://forge.coreweave.com/wandb/3213152090-bit/lerobot/runs/yidm7mzl?nw=nwuser3213152090)。

## 6. 真机部署与评估

### 6.1 部署设置

| 项目 | 设置 |
|---|---|
| Robot | Seeed B601-RS |
| 控制目标频率 | 30 Hz |
| 推理方式 | 同步 Action Chunk 推理 |
| Chunk Size | 100 |
| Temporal Ensembling | 未启用（`temporal_ensemble_coeff=None`） |
| 测试场景 | 不同电池初始位置 |

### 6.2 真机测试与迭代

在不同电池初始位置开展 25 次真机测试。首次测试成功率为 56%。

复盘失败案例后，发现失败主要集中在电池进入插槽阶段。分析认为腕部相机视角受限，影响了末端对电池与插槽相对位置的观测。

随后进行了两项部署迭代：

1. **调整腕部相机视角：** 将相机视角调整至更有利于观察末端与插槽接触区域的位置（历史记录为移至手腕下侧）。
2. **调整末端推压轨迹：** 将原来的直接推压调整为回拉后下压的轨迹。

调整后，记录的任务成功率由 **56% 提升至 72%**。

> 该提升来自失败复盘后的相机视角与末端执行轨迹调整。不能将全部提升归因于 ACT 模型重新训练或单纯的 Loss 下降。

## 7. 真机视频

本项目已完成 B601-RS 电池插槽任务真机演示视频采集。当前演示片段约 25.9 秒，分辨率 1280 × 720，视频帧率 30 FPS。

视频文件尚未提交到本仓库。公开展示时可将脱敏后的视频放入仓库的 `assets/` 目录，再在本节添加 GitHub 视频链接。

## 8. 实验结论

本项目完成了从单臂遥操作示教采集、LeRobot ACT 训练到真机部署评估的完整流程：

1. 使用固定遥控器的单臂任务设置，将数据采集随机性集中在电池初始位置。
2. 采集 50 条示教轨迹，覆盖遥控器同侧与异侧的电池摆放位置。
3. 使用 ResNet18 + Transformer + CVAE 的 ACT 策略训练 300K Steps。
4. 在 30 Hz 目标控制频率下进行同步动作块推理。
5. 通过 25 次真机测试和失败案例复盘，调整腕部相机视角及末端推压轨迹，将记录的成功率从 56% 提升至 72%。

## 9. 复现信息与待补充项

- [x] 数据集 Repo ID、Episode 数量与帧数
- [x] ACT 结构与主要模型参数
- [x] 训练步数、Batch Size、Optimizer 与学习率
- [x] W&B Run 链接
- [x] 30 Hz 真机部署设置
- [x] 失败复盘与成功率迭代记录
- [ ] 补充 LeRobot Git commit / 版本号
- [ ] 补充训练耗时与实际 GPU 型号
- [ ] 将真机视频上传至仓库 `assets/` 并添加链接
- [ ] 补充每轮 25 次测试的逐次结果与失败类型统计
