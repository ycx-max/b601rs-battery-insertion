# ACT — Single-Arm B601-RS Battery Insertion

基于 LeRobot ACT（Action Chunking with Transformers）的单臂电池插槽任务实验记录，涵盖遥操作示教、数据组织、模型训练、真机推理和实验分析。

## 1. Task and Robot Platform

### 1.1 Task

使用单臂机械臂完成电池抓取、移动、插槽对准和插入。

### 1.2 Hardware

| 项目 | 配置 |
|---|---|
| Follower | 1 × Seeed B601-RS |
| Leader | 1 × reBot Arm 102 |
| Framework | LeRobot |
| Policy | ACT |
| Task | Insert the battery into the slot |
| Camera Views | Front / Side |
| Image Resolution | 640 × 480 |
| Camera FPS | 30 FPS |
| Control Frequency | 30 Hz |

相机设备记录：Front：/dev/video4；Side：/dev/video10。

## 2. Demonstration Dataset

| 项目 | 记录 |
|---|---|
| Dataset Repo ID | seeed_rebot_b601_rs/test1012_2 |
| Episodes | 50 |
| Frames | 29,737 |
| Batch Size | 8 |
| Seed | 1000 |

数据由单臂遥操作采集，通过调整电池初始位置增加轨迹的空间分布。

### 2.1 Initial-Position Variation

| 初始位置 | Episodes | 采集方式 |
|---|---:|---|
| 左侧 | 30 | 每 5 条示教更换一次电池初始位置 |
| 右侧 | 20 | 每 5 条示教更换一次电池初始位置 |
| **Total** | **50** | 覆盖不同电池初始位置 |

## 3. ACT Model

ACT 将未来一段动作组织成 Action Chunk，通过 Transformer 学习观测与动作序列之间的映射。当前实验配置：

| 参数 | 配置 | 含义 |
|---|---:|---|
| use_vae | true | 使用 CVAE 潜变量 |
| latent_dim | 32 | 潜变量维度 |
| kl_weight | 10 | KL 正则项权重 |
| chunk_size | 100 | 预测动作块长度 |
| n_obs_steps | 1 | 输入观测时间步数 |
| n_action_steps | 100 | 每次执行动作步数 |
| n_encoder_layers | 4 | Transformer Encoder 层数 |
| n_decoder_layers | 1 | Transformer Decoder 层数 |
| temporal_ensemble_coeff | None | 关闭 Temporal Ensembling |
| Image Augmentation | Disabled | 关闭图像增强 |

Vision Backbone：ResNet18。

### 3.1 Action Chunk

动作块长度为 100，控制频率为 30 Hz。执行完整动作块的理论时长为：

$$
T = \frac{100}{30} \approx 3.33\ \mathrm{s}
$$

Temporal Ensembling 当前关闭。ACT 根据观测预测一段未来动作，再按照 n_action_steps 执行动作序列。

## 4. Training

### 4.1 Training Run

训练曲线：[W&B Run — act_rebot_test](https://forge.coreweave.com/wandb/3213152090-bit/lerobot/runs/yidm7mzl?nw=nwuser3213152090)

| 项目 | 配置 |
|---|---|
| Run Name | act_rebot_test |
| Run ID | yidm7mzl |
| Dataset | seeed_rebot_b601_rs/test1012_2 |
| Policy | ACT |
| Seed | 1000 |
| Device | CUDA |
| Batch Size | 8 |
| Training Steps | 300,000 |
| Checkpoint Save Frequency | 20,000 |

### 4.2 Optimizer

| 参数 | 配置 |
|---|---:|
| Optimizer | AdamW |
| Learning Rate | 1e-5 |
| Weight Decay | 1e-4 |
| Betas | (0.9, 0.999) |
| Gradient Clip Norm | 10 |

### 4.3 Equivalent Epochs

数据集包含 29,737 Frames，Batch Size 为 8，训练步数为 300,000。等效 Epoch 计算为：

$$
\mathrm{Epochs}
= \frac{300,000 \times 8}{29,737}
\approx 80.7
$$

### 4.4 Training Observations

W&B 记录包含 Training Loss、L1 Loss、KL Loss 和 Learning Rate。

- L1 Loss 随训练推进持续下降。
- KL Loss 接近 0，训练分析中重点关注 CVAE 潜变量的使用情况。
- 结合 Loss 曲线和真机执行表现分析模型训练效果。

## 5. Real-Robot Rollout

ACT 在 B601-RS 单臂平台上完成电池抓取与插入任务。针对真机失败案例进行复盘后，逐步优化相机视角、末端运动轨迹和动作时序。

### 5.1 Success Rate Improvement

| 实验阶段 | 调整内容 | Task Success Rate |
|---|---|---:|
| ACT 初始实验 | 初始相机视角与末端推压轨迹 | 56% |
| ACT 结构与轨迹优化 | 调整腕部相机视角，优化末端推压轨迹 | 72% |
| ACT + Temporal Ensembling | 在优化后的配置上启用 TE 模块 | **80%** |

### 5.2 Failure Analysis and Optimization

**腕部相机视角调整：** 复盘失败案例后调整腕部相机视角，改善末端执行器与电池、插槽区域的视觉观测。

**末端推压轨迹优化：** 调整末端接近插槽后的推压轨迹，改善电池对准和插入阶段的动作执行。

**Temporal Ensembling：** 在 ACT 推理过程中启用 TE，对不同时间预测得到的重叠动作进行加权融合，进一步提升动作连续性，实验成功率达到 80%。

### 5.3 Evaluation Record

| 评估项 | 记录内容 |
|---|---|
| Checkpoint | 模型路径与训练 Step |
| Initial Position | 左侧 / 右侧及具体摆放位置 |
| Grasp | 电池抓取情况 |
| Alignment | 插槽对准情况 |
| Insertion | 电池插入情况 |
| Completion Time | 完成任务所需时间 |
| Result | 成功 / 失败 |
| Failure Stage | 失败发生阶段 |

## 6. Experiment Analysis

### 6.1 Data Coverage

50 条示教覆盖左侧 30 条、右侧 20 条初始位置。每 5 条示教调整一次电池初始位置，使训练数据包含不同的电池与机械臂相对位置。

### 6.2 Latent Variable and KL

ACT 使用 CVAE 对示教动作中的多模态分布进行建模。KL Loss 接近 0 时，结合潜变量分布和动作预测结果分析潜变量是否被模型有效利用。

### 6.3 Action Chunk and Temporal Ensembling

ACT 一次预测多个未来动作。Temporal Ensembling 对不同时间预测得到的重叠动作进行加权融合，用于平滑动作执行。Action Chunk 长度和 Temporal Ensembling 是两个独立的配置维度。

## 7. Reproducibility Checklist

- [x] 记录单臂硬件组成
- [x] 记录 50 条示教与左右初始位置分布
- [x] 记录数据集 Repo ID 和 Frame 数量
- [x] 记录 ACT 主要模型参数
- [x] 记录训练步数、Batch Size 和 Optimizer
- [x] 保存 W&B 训练记录
- [x] 完成多次真机 Rollout
- [ ] 整理每次真机测试的成功 / 失败记录
- [ ] 归档真机演示视频并添加仓库预览链接
- [ ] 补充 LeRobot 代码版本与训练配置文件
- [ ] 对左右初始位置分别统计成功率
