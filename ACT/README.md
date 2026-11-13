# ACT — B601-RS Battery Insertion

基于 LeRobot ACT（Action Chunking with Transformers）的单臂电池插槽装配项目，涵盖遥操作示教采集、策略训练、真机部署与实验迭代。

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
| GPU | NVIDIA GeForce RTX 4090 |
| W&B Project | `lerobot` |
| W&B Run | `act_rebot_test` |
| Run ID | `yidm7mzl` |
| Seed | 1000 |

训练记录与曲线：[查看 W&B Run — act_rebot_test](https://forge.coreweave.com/wandb/3213152090-bit/lerobot/runs/yidm7mzl?nw=nwuser3213152090)

## 2. 单臂任务设计与示教数据采集

### 2.1 任务设置

原始 ACT 论文采用双臂操作完成电池插槽任务。本项目基于一台 Seeed B601-RS 机械臂搭建单臂操作平台，将遥控器固定在工作台上，并通过改变电池初始位置构建不同的任务初始状态。

### 2.2 电池初始位置分布

共采集 50 条专家示教轨迹：

| 位置类别 | Episodes | 摆放规则 |
|---|---:|---|
| 与遥控器同侧 | 30 | 每 5 条示教更换一次电池初始位置 |
| 与遥控器异侧 | 20 | 每 5 条示教更换一次电池初始位置 |
| **合计** | **50** | 覆盖 10 组电池相对位置 |

采集位置统计为左侧 30 Episodes、右侧 20 Episodes。

### 2.3 传感器与数据

- 双视角 RGB 相机：Front / Side。
- 图像分辨率：640 × 480。
- 相机帧率：30 FPS。
- 数据规模：50 Episodes，共 29,737 Frames。

## 3. ACT 模型配置

本次实验采用 LeRobot ACT 配置，结合 ResNet18 视觉编码器、Transformer 动作序列建模模块与 CVAE 潜变量，对示教动作序列进行学习。

| 参数 | 实际配置 | 说明 |
|---|---:|---|
| Vision Backbone | ResNet18 | ImageNet 预训练 |
| `use_vae` | `true` | 使用 CVAE 潜变量 |
| `latent_dim` | 32 | 潜变量维度 |
| `kl_weight` | 10 | KL 正则项权重 |
| `chunk_size` | 100 | 每次预测 100 步动作 |
| `n_obs_steps` | 1 | 输入观测时间步数 |
| `n_action_steps` | 100 | 每次执行 100 步动作 |
| `n_encoder_layers` | 4 | Transformer Encoder 层数 |
| `n_decoder_layers` | 1 | Transformer Decoder 层数 |
| `temporal_ensemble_coeff` | `None` | 关闭 Temporal Ensembling |
| `image_transforms.enable` | `false` | 关闭图像增强 |

### 3.1 Action Chunk 与推理方式

ACT 根据当前观测预测长度为 100 的动作序列（Action Chunk）。本次部署采用同步动作块推理，目标控制频率为 30 Hz，每个动作块对应约 3.33 秒的执行时间：

```text
100 / 30 ≈ 3.33 seconds
```

本次推理关闭 Temporal Ensembling（`temporal_ensemble_coeff=None`），执行完当前动作块后重新获取观测并预测下一段动作。

## 4. 模型训练

### 4.1 W&B 运行配置

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

模型可训练参数约为 51.6M。

按 Batch Size 8 计算，累计样本访问量为 2,400,000；以 29,737 帧估算，约相当于 80.7 个 Epoch：

```text
Equivalent Epochs
= 300000 × 8 / 29737
≈ 80.7
```

ACT 训练通过时间窗口从示教轨迹中构建动作序列样本，相邻样本包含重叠的未来动作片段。

### 4.4 训练启动命令

本次实验使用以下命令启动训练：

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

Batch Size、随机种子、Checkpoint 保存频率及优化器参数见本节配置表和 W&B 运行记录。

## 5. 训练结果分析

W&B 记录了 `train/loss`、`train/l1_loss`、`train/lr`、`train/steps` 等训练指标。

- **L1 Loss：** 从约 6–7 持续下降至约 0.02–0.03，模型对示教动作序列的拟合逐步收敛。
- **KL Loss：** 训练过程中接近 0，呈现潜变量后验向先验分布收缩的现象，可结合潜变量分布及真机表现进一步分析。
- **整体训练趋势：** 动作预测误差持续降低，训练过程趋于稳定。

完整训练曲线见 [W&B Run](https://forge.coreweave.com/wandb/3213152090-bit/lerobot/runs/yidm7mzl?nw=nwuser3213152090)。

## 6. 真机部署与实验评估

### 6.1 部署设置

| 项目 | 设置 |
|---|---|
| Robot | Seeed B601-RS |
| 控制目标频率 | 30 Hz |
| 推理方式 | 同步 Action Chunk 推理 |
| Chunk Size | 100 |
| Temporal Ensembling | 关闭 |
| 测试场景 | 不同电池初始位置 |

### 6.2 真机测试与策略迭代

在不同电池初始位置下开展 25 次真机测试，基线成功率为 56%。失败主要发生在电池对准插槽及插入阶段，实验复盘发现腕部相机视角对末端与插槽相对位置的观测存在局限。

针对上述问题，进行了两项优化：

1. **相机视角调整：** 将腕部相机移动至手腕下侧，改善末端执行区域及插槽入口的视觉观测。
2. **末端轨迹优化：** 将直接向前推压调整为回拉后下压的动作轨迹，改善电池与插槽之间的对准和接触过程。

完成上述调整后，真机任务成功率由 **56% 提升至 72%**，电池插入阶段的执行稳定性得到改善。

## 7. 真机演示视频

本项目完成了 B601-RS 电池抓取与插槽装配的真机演示。当前演示视频参数如下：

| 项目 | 参数 |
|---|---|
| 时长 | 约 25.9 秒 |
| 分辨率 | 1280 × 720 |
| 帧率 | 30 FPS |

视频素材可归档至仓库 `assets/` 目录，并在此处补充在线预览链接。

## 8. 实验总结

本项目基于 Seeed B601-RS 与 LeRobot 完成 ACT 单臂电池插槽装配任务，建立了从遥操作示教、数据集构建、策略训练到真机部署评估的完整流程。

1. **任务与数据构建：** 搭建单臂遥操作平台，固定遥控器位置，通过改变电池初始位置采集 50 条专家示教轨迹，共 29,737 帧。
2. **策略训练：** 基于 ResNet18、Transformer 和 CVAE 架构训练 ACT 模型，使用 Batch Size 8 完成 300K Steps 训练。
3. **真机部署：** 在 30 Hz 目标控制频率下进行 Action Chunk 推理，每次预测并执行 100 步动作。
4. **实验迭代：** 通过 25 次真机测试定位插入阶段的主要问题，并针对腕部相机视角和末端执行轨迹进行优化，成功率由 56% 提升至 72%。

## 9. 后续实验与资料归档

- [x] 数据集 Repo ID、Episode 数量与帧数
- [x] ACT 模型结构与主要参数
- [x] 训练步数、Batch Size、Optimizer 与学习率
- [x] W&B 训练记录与曲线
- [x] 真机部署设置与成功率迭代
- [x] 真机演示视频信息
- [ ] 补充 LeRobot Git Commit / 版本号
- [ ] 补充训练耗时与 GPU 型号信息
- [ ] 将真机视频归档至仓库 `assets/` 并添加预览链接
- [ ] 整理 25 次测试的逐次结果与失败类型
