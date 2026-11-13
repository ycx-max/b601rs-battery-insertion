# B601-RS Battery Insertion

基于 Seeed B601-RS 单臂机械臂与 LeRobot 的真实机器人操作项目，围绕电池抓取、移动、对准和插入插槽任务，开展模仿学习与 Vision-Language-Action（VLA）策略的训练和真机部署。

## 真机演示

[![B601-RS 机械臂电池插槽装配真机演示](./assets/real-robot-demo-preview.webp)](./assets/real-robot-demo.mp4)

[观看原始 MP4](./assets/real-robot-demo.mp4)

### 真机演示关键帧

![电池插槽装配演示的 8 个关键帧（0.0–25.8 秒）](./assets/real-robot-demo-frames.png)

## 1. Project Overview

| 项目 | 配置 |
|---|---|
| Task | Insert the battery into slot |
| Follower | 1 × Seeed B601-RS |
| Leader | 1 × reBot Arm 102 |
| Framework | LeRobot |
| Research Focus | Imitation Learning、VLA Fine-Tuning、Real-Robot Evaluation |

## 2. Policy Implementations

| Policy | Directory | 内容 |
|---|---|---|
| ACT | [ACT/](./ACT/) | 单臂遥操作示教、Action Chunking、模型训练与真机部署 |
| π0.5 | [pi0.5/](./pi0.5/) | LoRA、Expert-Only、Full Fine-Tuning 与 VLA 真机适配 |
| GR00T | [groot/](./groot/) | 后续模型适配与实验记录 |

## 3. Key Real-Robot Results

当前单臂电池插槽任务的阶段性真机成功率：

| Policy / Experiment | 优化方式 | Success Rate |
|---|---|---:|
| ACT | 初始实验 | 56% |
| ACT | 调整腕部相机视角与末端推压轨迹 | 72% |
| ACT + TE | 启用 Temporal Ensembling | 80% |
| π0.5 LoRA r32 | Rank = 32 | 36% |
| π0.5 Expert-Only | 仅训练 Action Expert | 52% |
| π0.5 Full Fine-Tuning | 全参数微调 | 76% |
| π0.5 Full FT + RTC | 全参数微调并启用 RTC | **92%** |

### ACT 优化过程

通过复盘 ACT 真机失败案例，调整腕部相机视角和末端推压轨迹，成功率由 56% 提升至 72%；进一步启用 Temporal Ensembling（TE）后，成功率达到 80%。

### π0.5 微调与推理优化

π0.5 不同微调策略的真机成功率分别为：LoRA r32 36%、Expert-Only 52%、Full Fine-Tuning 76%。在 Full Fine-Tuning 模型上进一步结合 RTC，成功率达到 92%。

## 4. Research Workflow

```text
Single-Arm Robot Setup
        ↓
Teleoperation Demonstrations
        ↓
Dataset Inspection & Preprocessing
        ↓
Policy Training / Fine-Tuning
        ↓
Checkpoint Selection
        ↓
Real-Robot Rollout
        ↓
Failure Analysis & Data Iteration
```

## 数据采集与 LeRobot 数据格式

### 数据采集设置

操作者使用 reBot Arm 102（Leader）遥操作 Seeed B601-RS（Follower），采集电池抓取、移动、插槽对准和插入的示教轨迹。一次示教对应一个 episode，包含该次操作的连续时间步。

| 项目 | ACT | π0.5 |
|---|---|---|
| LeRobot Dataset Repo ID | `seeed_rebot_b601_rs/test1012_2` | `seeed_rebot_b601_rs/test_1102` |
| 示教轨迹数 | 50 条 | 50 条 |
| 总帧数 | 29,737 | 32,666 |
| 相机视角 | 正视（Front）、侧视（Side） | 正视（Front）、侧视（Side） |
| 图像分辨率与相机帧率 | 640 × 480，30 FPS | 640 × 480，30 FPS |
| 其他已记录配置 | 控制频率 30 Hz | 视频读取后端 PyAV |

50 条真机示教按**电池的初始摆放位置**均分，左侧 25 条（50%），右侧 25 条（50%）。每采集 5 条示教调整一次初始摆放位置，以覆盖同一侧的不同位置。

### LeRobot 数据格式

LeRobot 按时间步关联相机观测、机器人状态和示教动作，并用 episode 索引将连续时间步归入同一条轨迹。

| 数据 | 常见字段或索引 | 含义 |
|---|---|---|
| 多视角图像 | `observation.images.<camera_key>` | 按相机读取当前时间步的正视与侧视画面 |
| 机器人状态 | `observation.state` | 当前时间步记录的机器人状态向量 |
| 示教动作 | `action` | 当前时间步的示教控制指令，用于动作监督 |
| 任务 | `task_index` | 查表取得自然语言任务描述。π0.5 数据记录中，`0` 对应 `Insert the battery into slot` |
| 轨迹与时间索引 | `episode_index`、`frame_index`、`timestamp` | 标识轨迹归属、轨迹内帧序号和采样时间 |

在 LeRobotDataset v3.0 的 Parquet/MP4 布局中，`data/` 下的 Parquet 分片存放状态、动作和索引，`videos/` 按相机存放 MP4 视频分片，`meta/episodes/` 记录轨迹边界和分片定位信息。训练端通过元数据定位同一轨迹中的数值记录和各路视频。任务映射存于 `meta/tasks.parquet`，数据版本、字段名称、维度、数据类型和帧率以对应数据集的 `meta/info.json` 为准。[LeRobot 官方格式说明](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

π0.5 训练读取当前时刻的两路相机观测、机器人状态和任务文本，并用示教中的未来动作序列作为监督。预处理将归一化后的机器人状态离散化，再与任务文本拼成提示词。训练环境为 Conda `lerobot`，使用 2 张 NVIDIA RTX 4090。[ACT 采集记录](./ACT/README.md) · [π0.5 数据预处理与训练配置](./pi0.5/README.md) · [π0.5 实验记录](./pi0.5/EXPERIMENTS.md)

## 5. Repository Structure

```text
b601rs-battery-insertion/
├── README.md
├── ACT/
│   └── README.md
├── pi0.5/
│   ├── README.md
│   └── EXPERIMENTS.md
├── groot/
│   └── README.md
├── .gitignore
└── LICENSE
```

## 6. Experiment Records

每组实验记录以下内容：

- **Robot & Task：** 机械臂型号、相机布局、控制频率和任务定义。
- **Dataset：** Repo ID、Episode 数量、Frame 数量、任务文本和数据划分。
- **Model：** Base Checkpoint、模型结构和参数更新范围。
- **Training：** GPU、精度、Batch Size、Steps、Epoch、Optimizer 和 Scheduler。
- **Inference：** Checkpoint、Action Chunk、推理频率和控制接口。
- **Evaluation：** 测试次数、成功次数、成功率、完成时间和失败阶段。

## 7. References

- [LeRobot](https://github.com/huggingface/lerobot)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
- [π0.5 Real-Robot Fine-Tuning: Data, RTC and Evaluation](https://www.cnblogs.com/ZikangZhou/p/23390824/pi05-real-robot-finetuning-data-rtc)
