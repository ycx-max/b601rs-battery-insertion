# B601-RS Battery Insertion

基于 Seeed B601-RS 与 LeRobot 的真实机器人操作项目，围绕电池抓取、移动、对准和插槽插入任务，记录模仿学习与 Vision-Language-Action（VLA）策略的训练、部署和真机评估。

本仓库按策略路线组织实验文档，重点保留数据版本、模型配置、训练预算、推理设置和真机结果，方便复现与横向比较。

## 1. Project Overview

**Task:** Battery grasping and slot insertion  
**Robot:** Seeed B601-RS  
**Framework:** LeRobot  
**Research focus:** Imitation Learning, VLA Fine-Tuning, Real-Robot Evaluation

| Policy | Directory | Project Status |
|---|---|---|
| ACT | [ACT/](./ACT/) | 已建立 ACT 真机示教、训练与部署实验记录 |
| π0.5 | [pi0.5/](./pi0.5/) | 正在整理 LoRA、Expert-Only、Full Fine-Tuning 与真机评估 |
| GR00T | [groot/](./groot/) | 后续适配与实验记录 |

## 2. Research Workflow

整个项目围绕一条完整的机器人学习链路展开：

```text
Robot & Task Setup
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

不同策略共享同一类机器人任务，但数据版本、输入表示、动作空间和训练目标可能不同。对比时应明确标注具体实验配置，不能只按模型名称直接比较结果。

## 3. Repository Structure

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

## 4. π0.5 Experiments

π0.5 是当前重点整理的 VLA 微调路线。当前使用的数据集为：

| Item | Value |
|---|---|
| Dataset Repo ID | `seeed_rebot_b601_rs/test_1102` |
| Episodes | 50 |
| Frames | 32,666 |
| Task label in `tasks.parquet` | `slot battery` |
| Training environment | Conda `lerobot` |
| GPU platform | 2 × RTX 4090（双卡实验） |

实验包含 LoRA r16、LoRA r32、Expert-Only 和 Full Fine-Tuning 等不同参数更新策略。具体训练配置、Prompt 构造、Epoch 换算与待完成的真机评估见 [π0.5 实验文档](./pi0.5/README.md) 和 [实验记录表](./pi0.5/EXPERIMENTS.md)。

## 5. Experiment Reporting Convention

每组实验尽量保留以下信息：

1. **Task & Robot：** 任务定义、机械臂型号、相机布局和控制频率。
2. **Dataset：** Repo ID、Episode 数量、Frame 数量、任务文本和数据划分。
3. **Model：** Base Checkpoint、模型结构和参数更新范围。
4. **Training：** GPU、精度、Batch Size、Global Batch Size、Steps、Epoch、Optimizer 和 Scheduler。
5. **Inference：** Checkpoint、Action Chunk、推理频率、动作队列和控制接口。
6. **Evaluation：** 测试次数、成功次数、成功率、完成时间和失败阶段。

训练 Loss 用于观察优化过程；最终任务表现以统一测试协议下的真机 Rollout 为准。

## 6. Data and Model Artifact Policy

仓库主要保存代码说明、实验配置和分析文档。原始机器人数据集、完整模型 Checkpoint、训练缓存和较大的真机视频不默认提交到 Git。

如需分享数据或模型，应使用独立的数据集 / 模型链接，并在公开前检查视频、日志和配置中是否包含访问令牌、密钥、个人信息或其他不宜公开的内容。

## 7. References

- [LeRobot](https://github.com/huggingface/lerobot)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
- [π0.5 Real-Robot Fine-Tuning: Data, RTC and Evaluation](https://www.cnblogs.com/ZikangZhou/p/23390824/pi05-real-robot-finetuning-data-rtc)
