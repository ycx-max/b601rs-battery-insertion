# ACT — Dual-Arm B601-RS Battery Insertion

基于 LeRobot ACT（Action Chunking with Transformers）的双臂电池插槽任务实验记录，覆盖遥操作示教、数据组织、模型训练、真机推理和实验复盘。

## 1. Task and Robot Platform

### 1.1 Task

机器人需要完成电池抓取、双臂协同操作与插槽插入。评估关注完整任务链路，而不是只判断某个机械臂是否完成抓取。

### 1.2 Hardware

| 项目 | 配置 |
|---|---|
| Follower | 2 × Seeed B601-RS |
| Leader | 2 × reBot Arm 102 |
| Control | Dual-arm teleoperation |
| Framework | LeRobot |
| Policy | ACT |
| Task | Insert the battery into the slot |
| Camera Views | Front / Side |
| Image Resolution | 640 × 480 |
| Camera FPS | 30 FPS |
| Target Control Frequency | 30 Hz |

相机设备记录：

- Front：`/dev/video4`
- Side：`/dev/video10`

相机设备编号与 USB 接口和系统枚举顺序有关，换机器后应重新检查设备映射。

## 2. Demonstration Dataset

### 2.1 Dataset Summary

| 项目 | 记录 |
|---|---|
| Dataset Repo ID | `seeed_rebot_b601_rs/test1012_2` |
| Episodes | 50 |
| Frames | 29,737 |
| Batch Size | 8 |
| Seed | 1000 |

数据由双臂遥操作采集。通过改变电池初始位置，增加训练轨迹对不同空间布局的覆盖。

### 2.2 Initial-Position Variation

50 条示教包含左侧和右侧两类初始位置：

| 初始位置 | Episodes | 采集方式 |
|---|---:|---|
| 左侧 | 30 | 每 5 条示教更换一次电池初始位置 |
| 右侧 | 20 | 每 5 条示教更换一次电池初始位置 |
| **Total** | **50** | 覆盖不同的电池—遥操作平台相对位置 |

采集设计还包含两类相对布局：电池位于遥操作器同侧，以及位于遥操作器异侧。这样可以减少策略只适应单一初始摆放位置的风险。

这里的“位置多样性”指示教开始时电池相对机器人和遥操作平台的位置变化，不代表已经覆盖所有可能的工作空间状态。

## 3. ACT Model

ACT 将未来一段动作组织成 Action Chunk，并通过 Transformer 学习观测与动作序列之间的映射。当前实验配置记录如下：

| 参数 | 配置 | 含义 |
|---|---:|---|
| Vision Backbone | ResNet18 | 图像特征提取 |
| `use_vae` | true | 使用 CVAE 潜变量 |
| `latent_dim` | 32 | 潜变量维度 |
| `kl_weight` | 10 | KL 正则项权重 |
| `chunk_size` | 100 | 预测动作块长度 |
| `n_obs_steps` | 1 | 输入观测时间步数 |
| `n_action_steps` | 100 | 每次执行动作步数 |
| `n_encoder_layers` | 4 | Transformer Encoder 层数 |
| `n_decoder_layers` | 1 | Transformer Decoder 层数 |
| `temporal_ensemble_coeff` | None | 当前配置关闭 Temporal Ensembling |
| Image Augmentation | Disabled | 关闭图像增强 |

### 3.1 Action Chunk and Execution

当前记录的动作块长度为 100，控制目标频率为 30 Hz。若连续执行完整动作块，对应的时间长度约为：

\[
T=\frac{100}{30}\approx 3.33\ \mathrm{s}
\]

本次配置关闭 Temporal Ensembling。动作块执行与下一次观测、推理的衔接由 LeRobot 的推理流程决定。动作块长度不等于机器人必须连续执行 100 步后才允许停止；部署时仍应结合安全条件和实际控制逻辑。

## 4. Training

### 4.1 Training Run

训练曲线记录：

[W&B Run — act_rebot_test](https://forge.coreweave.com/wandb/3213152090-bit/lerobot/runs/yidm7mzl?nw=nwuser3213152090)

| 项目 | 配置 |
|---|---|
| Run Name | `act_rebot_test` |
| Run ID | `yidm7mzl` |
| Dataset | `seeed_rebot_b601_rs/test1012_2` |
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
| Learning Rate | `1e-5` |
| Weight Decay | `1e-4` |
| Betas | `(0.9, 0.999)` |
| Gradient Clip Norm | 10 |

### 4.3 Equivalent Epochs

数据集包含 29,737 Frames，Batch Size 为 8，训练步数为 300,000。按累计样本访问量估算：

\[
\mathrm{Equivalent\ Epochs}
=\frac{300,000\times 8}{29,737}
\approx 80.7
\]

这里的 Epoch 是等效样本访问次数，不一定意味着数据加载器严格、无重复地完整遍历了数据集 80.7 次。

### 4.4 Training Observations

当前 W&B 记录包含 Training Loss、L1 Loss、KL Loss 和 Learning Rate 等指标。

- L1 Loss 在训练过程中持续下降，反映动作预测误差逐步减小。
- KL Loss 接近 0，提示需要进一步检查 CVAE 后验分布是否充分利用潜变量。
- 训练曲线用于观察优化过程；是否真正改善任务表现，仍需要通过真机 Rollout 判断。

完整训练曲线以 [W&B Run](https://forge.coreweave.com/wandb/3213152090-bit/lerobot/runs/yidm7mzl?nw=nwuser3213152090) 为准。

## 5. Real-Robot Rollout

训练完成后，模型已部署到 B601-RS 双臂平台进行真机测试，并完成多次成功执行。真机视频用于观察完整动作过程，包括抓取、双臂协同、移动和插入。

当前文档不将单次视频或未统一测试条件下的多次运行换算成统计成功率。后续评估建议记录：

| 评估项 | 记录内容 |
|---|---|
| Checkpoint | 模型路径与训练 Step |
| Initial Position | 左侧 / 右侧及具体摆放位置 |
| Grasp | 电池是否稳定抓取 |
| Coordination | 双臂动作是否按预期配合 |
| Insertion | 是否完成插入 |
| Completion Time | 完成任务所需时间 |
| Result | 成功 / 失败 |
| Failure Stage | 首次失败阶段 |

建议每个 Checkpoint 在相同初始位置集合下进行多次测试，并同时报告成功次数与总测试次数。

## 6. Experiment Notes

### 6.1 Data Coverage

50 条示教中包含左右两侧的初始位置变化。后续可以进一步扩大位置变化范围，并单独检查模型对未见初始位置的泛化表现。

### 6.2 Latent Variable and KL

当 KL Loss 长期接近 0 时，需要结合潜变量分布、重建误差和真机行为判断是否存在后验坍缩。不能只根据 KL 数值就认定模型一定发生了后验坍缩。

### 6.3 Action Chunk and Temporal Ensembling

ACT 一次预测多个未来动作，执行时只使用当前配置指定的动作步数。Temporal Ensembling 则通过融合不同时间预测得到的重叠动作，缓解动作抖动。两者是不同机制，应分别做对照实验。

## 7. Reproducibility Checklist

- [x] 记录双臂硬件组成
- [x] 记录 50 条示教与左右初始位置分布
- [x] 记录数据集 Repo ID 和 Frame 数量
- [x] 记录 ACT 主要模型参数
- [x] 记录训练步数、Batch Size 和 Optimizer
- [x] 保存 W&B 训练记录
- [x] 完成多次真机 Rollout
- [ ] 整理每次真机测试的成功 / 失败记录
- [ ] 将真机演示视频归档并添加仓库预览链接
- [ ] 补充 LeRobot 代码版本与训练配置文件
- [ ] 对左右初始位置分别统计成功率

