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
| Task | Insert the battery into slot |
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
| 左侧 | 25 | 每 5 条示教更换一次电池初始位置 |
| 右侧 | 25 | 每 5 条示教更换一次电池初始位置 |
| **Total** | **50** | 覆盖不同电池初始位置 |

## 3. ACT Model

ACT 将未来一段动作组织成 Action Chunk，通过 Transformer 学习观测与动作序列之间的映射。当前实验配置：

| 参数                      |         配置 | 含义                            |
| ----------------------- | ---------: | ----------------------------- |
| use_vae                 |       true | 使用 CVAE 潜变量                   |
| latent_dim              |         32 | 潜变量维度                         |
| kl_weight               |         10 | KL 正则项权重                      |
| chunk_size              |        100 | 预测动作块长度                       |
| n_obs_steps             |          1 | 输入观测时间步数                      |
| n_action_steps          |        100 | 每次执行动作步数                      |
| n_encoder_layers        |          4 | Transformer Encoder 层数        |
| n_decoder_layers        |          1 | Transformer Decoder 层数        |
| temporal_ensemble_coeff | None（初始配置） | 初始 ACT 实验关闭 TE；后续单独开展 TE 对照实验 |
| Image Augmentation      |   Disabled | 关闭图像增强                        |

Vision Backbone：ResNet18。

### 3.1 Action Chunk

动作块长度为 100，控制频率为 30 Hz。执行完整动作块的理论时长为：

$$
T = \frac{100}{30} \approx 3.33\ \mathrm{s}
$$

Temporal Ensembling 当前关闭。ACT 根据观测预测一段未来动作，再按照 n_action_steps 执行动作序列。

## 4. Training

### 4.1 Training Run



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



## 5. Experiment Analysis
ACT 在 B601-RS 单臂平台上完成电池抓取与插入任务。针对真机失败案例进行复盘后，逐步优化相机视角、末端运动轨迹和动作时序。

### 5.1 Success Rate Improvement

| 实验阶段 | 调整内容 | Task Success Rate |
|---|---|---:|
| ACT 初始实验 | 初始相机视角与末端推压轨迹 | 14/25（56%） |
| ACT 结构与轨迹优化 | 调整腕部相机视角，优化末端推压轨迹 | 18/25（72%） |
| ACT + Temporal Ensembling | 在优化后的配置上启用 TE 模块 | **20/25（80%）** |

### 5.2 Failure Analysis and Optimization

**腕部相机视角调整：** 复盘失败案例后调整腕部相机视角，改善末端执行器与电池、插槽区域的视觉观测。

**末端推压轨迹优化：** 调整末端接近插槽后的推压轨迹，改善电池对准和插入阶段的动作执行。

### 5.3 n_action_steps

一开始设置n_action_steps=50，chunk_size=100，发现机械臂抖动明显。ACT 论文明确指出，这种“每 k 步突然纳入新观测”的机制，本身就可能引起**机器人运动的抖动（jerky motion）**。在真机上，这种接缝抖动通常表现为**有规律的、周期性的突跳**，其间隔正好等于 `n_action_steps / 控制频率`。把 `n_action_steps` 从 100 减到 50，等于把这种突跳的频率**提高了一倍**，体感上自然会觉得“抖动严重了很多#[Action Chunking Explained: Chunk Size, Executed Steps, Jitter | AY-Robots](https://www.ay-robots.com/blog/en/action-chunking-explained)

### 5.4 Latent Variable and KL

ACT 使用 CVAE 对示教动作中的多模态分布进行建模。KL Loss 接近 0 时，结合潜变量分布和动作预测结果分析潜变量是否被模型有效利用。一开始的实验设置中，把电池初始位置都放在了遥控器左侧，这使得人类专家示教时的动作风格没有太大的变化，训练出来的曲线中kl_loss几乎为零（0.00002）、远小于重建损失（0.02），ACT训练很可能出现了后验坍缩（Posterior Collapse），也就是CVAE学到的隐变量 \(z\) 逐渐失去了对动作预测的贡献，但是也能有较为不错的成功率。当示教按照左右划分后，cvae开始起作用。

### 5.5 Action Chunk and Temporal Ensembling

ACT 一次预测多个未来动作。Temporal Ensembling 对不同时间预测得到的重叠动作进行加权融合，用于平滑动作执行。Action Chunk 长度和 Temporal Ensembling 是两个独立的配置维度。
如果要启用，需要设置`temporal_ensemble_coeff`的值，原文默认0.01，同时要设置n_action_steps=1,推理代价变大，但是act较为轻量，未在真机发现明显延迟或卡顿。

这个系数，控制的是融合时对新旧预测的权重衰减速度。系数越小，衰减越慢，历史预测的权重越高，平滑作用越强，但可能引入响应延迟；系数越大，则更信任最新预测，平滑作用减弱，但响应更灵敏

## 6.Problems to be sloved

### 6.1 transformer decoder
 代码src\lerobot\policies\act\configuration_act.py156行处，transformer decoder 论文设置为7层，但ACT源代码在这一处有一个bug，只取了第一层的输出，故lerobot中直接把decoder设置为了1层。
### 6.2 CVAE
推理时令z=0，是为了得到确定性的策略输出，**但并不保证该输出一定是最优或某一种有效动作模式**。是否有方法可以对动作轨迹进行评价，并且筛选出潜在动作空间中最优的模式？
### 6.3  ACT为什么能解决单步BC的误差累计问题？
>需要区分纯Action Chunking和结合Temporal Ensembling的ACT。

对于纯Action Chunking，假设任务需要600个控制时间步，动作块长度为100，那么单步BC需要600次策略预测，而纯AC只需要6次动作块预测。这相当于缩短了有效策略决策时域，减少了执行偏移状态反复进入策略、影响下一步预测的机会，从而缓解一部分误差累积问题。

但纯AC存在局限：
-  动作预测本身存在误差，形成不合理轨迹；
    
- 实际执行状态偏离预测时，后续动作缺乏及时纠偏。

因此ACT引入Temporal Ensembling。它让模型在每个时间步重新预测未来一段动作，再将不同历史时刻对当前动作的重叠预测进行指数加权平均。

这使当前动作既能够利用历史预测所保留的运动连续性，也能通过最新预测引入实时观测信息，从而改善动作平滑性，并在一定条件下降低单次预测误差的影响。

对于AC+TE，减少策略调用次数这一解释不再成立，但动作分块学习时间结构的优势仍然存在，TE还通过融合历史与当前预测，降低对单次动作预测的依赖。

需要注意，TE并不能完全消除误差累积，也不能保证历史预测一定比最新预测准确。它可能减小某一次预测异常对最终执行的影响。但如果多个预测存在相同的系统性偏差，加权平均也无法消除这种偏差

除此之外，动作分块还有独立于TE的重要价值：它能够直接学习连续动作序列的时间结构，例如一段操作中的接近、减速、暂停和重新运动。对于人工示教中存在的停顿和阶段性动作，这种序列建模比单帧、单步BC更有优势(eg. 热油炒菜例子单步BC不能理解停顿的过程是 刚开始等待；已经等待了20步；即将结束等待。)
