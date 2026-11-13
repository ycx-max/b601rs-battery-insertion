# B601-RS Battery Insertion

基于 Seeed B601-RS 机械臂与 LeRobot，复现并比较不同模仿学习 / Vision-Language-Action（VLA）策略在电池抓取与插槽装配任务上的表现。

**核心任务：** `Insert the battery into the slot`

## 项目目标

本仓库用于集中记录同一类真机操作任务在不同 Policy 下的实现、训练配置、数据集信息、实验过程和评估结果，便于复现、对比与持续迭代。

当前按 Policy 类型组织：

| Policy | 目录 | 当前记录 |
|---|---|---|
| ACT | [ACT/](./ACT/) | 已记录 50 条真机示教、300K Steps 训练配置及 25 次真机测试的迭代结果 |
| π0.5 | [pi0.5/](./pi0.5/) | 已记录双 RTX 4090 DDP 全参数微调实验 |
| GR00T | [groot/](./groot/) | 预留目录，后续补充实现与实验记录 |

## 仓库结构

```text
b601rs-battery-insertion/
├── README.md
├── ACT/
│   └── README.md
├── pi0.5/
│   └── README.md
├── groot/
│   └── README.md
├── .gitignore
└── LICENSE
```

## 实验平台

- **Robot：** Seeed B601-RS
- **Framework：** LeRobot
- **Task：** Battery grasping and slot insertion
- **Policies：** ACT、π0.5、GR00T

各 Policy 的数据集、模型配置、训练命令、Checkpoint 和评估结果分别记录在对应目录中。

## 实验记录约定

为保证不同实验之间可比较，每份实验记录尽量包含：

1. 机器人平台与任务定义
2. 数据集 Repo ID、Episode 数量和帧数
3. Base Model 与 Policy 配置
4. GPU、分布式训练和数值精度
5. Batch Size、训练步数、优化器与学习率配置
6. 训练日志与模型版本
7. 真机 Rollout 设置、成功率与失败案例

**说明：** 仓库以代码、配置和实验文档为主。原始机器人数据集、视频和大型模型 Checkpoint 不直接提交到 Git；如需共享，应单独提供数据集或模型链接，并确认文件中不包含密钥、访问令牌等敏感信息。
