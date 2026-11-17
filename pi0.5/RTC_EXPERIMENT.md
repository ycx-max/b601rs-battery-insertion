# π0.5 RTC 真机调参与调试记录

本文记录 π0.5 在 Seeed B601-RS 电池插槽任务上启用 Real-Time Chunking（RTC）时的真机调参、延迟测量、问题定位和代码修复过程。

> **记录日期：** 2026-11-16
>
> **调试 checkpoint：** Full Fine-Tuning 62,000-step checkpoint
>
> **任务：** 电池抓取、对准并插入插槽
>
> **相机：** Front / Side，640 × 480，30 FPS
>
> **说明：** 本文的延迟与队列数据来自 62k checkpoint 的工程调试，不替代 80k checkpoint 的 25 次最终成功率评测。仓库中 84%（21/25）的结果仍以 [EVALUATION.md](./EVALUATION.md) 为准。

## 1. 调试目标

初始 RTC rollout 出现以下现象：

- 新旧动作块交界处偶发抖动；
- 机械臂在接近目标或完成插入动作时出现回弹；
- 50 Hz 下推理期间会跨越较多控制步；
- 日志频繁出现动作队列估算延迟与实际消费步数不一致。

本轮调试的目标是：

1. 测量真机 RTC 的模型推理和端到端延迟；
2. 对比 50 Hz 与 30 Hz 控制频率；
3. 检查 `queue_threshold`、`execution_horizon` 和 RTC 引导权重；
4. 定位回弹是否来自延迟估计、RTC 权重或动作块对齐；
5. 保留 rollout 视频与可复核的延迟日志。

## 2. 参数基线与调参过程

### 2.1 默认值、变化与最终配置

本轮并不是直接选定最终参数，而是先从 LeRobot RTC 的默认/初始配置开始，在测量并修复延迟统计和动作队列对齐问题后，再逐项调参：

| 参数 | LeRobot 默认/本轮初始值 | 调整后 | 变化原因 |
|---|---:|---:|---|
| `action_queue_size_to_get_new_actions`（`queue_threshold`） | 30 | **20** | 修复延迟与队列对齐后，依据稳态尾延迟缩小提前量，减少过早重新推理和动作块切换 |
| `rtc.execution_horizon` | 10 | **10** | 处于建议的 8–12 步初始范围，暂不与其他变量同时改变 |
| `rtc.max_guidance_weight` | 10.0 | **5.0** | 10.0 下旧轨迹约束偏强，降低后减弱拉回、回弹和块边界抖动 |
| `rtc.prefix_attention_schedule` | EXP | **EXP** | 保留指数衰减，使旧动作前缀的影响逐步减弱 |
| `fps` | 50 Hz（首轮测试） | **30 Hz** | 毫秒延迟基本不变，但推理跨越的控制步数由约 8 步降至 5 步 |
| RTC 延迟估计 | 历史最大值，包含冷启动 | **guided P95** | 避免 397–406 ms 冷启动永久污染 RTC；P99 同时用于尾延迟监测 |
| `rtc_bootstrap_latency_s` | 无独立稳态先验 | **0.16 s** | 首个 guided chunk 在尚无稳态样本时采用 160 ms 先验 |

因此，本轮关键参数变化链为：

```text
初始：queue_threshold=30, execution_horizon=10, max_guidance_weight=10
  ↓ 测量端到端 P95/P99，发现冷启动污染和队列对齐 bug
修复：排除冷启动、采用 guided P95、按实际消费步数合并动作块
  ↓
队列：queue_threshold 30 → 20
  ↓
引导：max_guidance_weight 10 → 5
最终：queue_threshold=20, execution_horizon=10, max_guidance_weight=5
```

### 2.2 四步调参方法

#### 第一步：先测端到端推理延迟

使用端到端 P95/P99 latency，而不是只看模型 forward 时间。端到端统计包含观测读取、预处理、同步模型推理和后处理，更接近 actor 在推理期间实际消费动作的时间。

本轮正是在这一步发现了程序问题：首次 CUDA / 模型冷启动被写入历史最大延迟，且根据 `ceil(latency × fps)` 估算的步数有时比 actor 实际消费步数多 1。若不先修复这些问题，后续对 RTC 参数的判断会建立在错误的延迟和动作块对齐之上。

#### 第二步：调整 `queue_threshold`

以 P95 延迟对应的控制步数加安全余量作为初始参考：

$$
Q \gtrsim d_{P95} + m
$$

其中 $Q$ 为 `queue_threshold`，$d_{P95}$ 为端到端 P95 延迟换算出的控制步数，$m$ 可先取 3–5 步。当前 LeRobot 还会根据实际推理耗时自适应提前请求，因此该公式用于确定起点，不是硬性限制。

- $Q$ 过小：可能来不及生成新动作，导致队列耗尽；
- $Q$ 过大：会在队列仍较长时过早重新推理，增加不必要的推理和动作块切换频率。

30 Hz 修复后，端到端 P95 为 159.02 ms，对应：

```text
d_P95 = ceil(0.15902 s × 30 Hz) = 5 steps
d_P95 + m = 8–10 steps
```

本轮将 LeRobot 默认值 **30** 下调为 **20**。20 仍是相对保守的真机配置；在 `execution_horizon=10`、延迟 5 步时，相对 `5 + 10 = 15` 仍保留 5 步队列余量，但触发时间晚于 30，可减少过早重新推理和频繁切换。

#### 第三步：调整 `execution_horizon`

- 过小：新旧 action chunk 的重叠约束不足，块边界可能不连续；
- 过大：旧轨迹影响范围过长，降低策略对新观测的响应能力；
- 可从 8–12 步作为初始实验范围。

本轮保持 **10 步**不变，以避免在验证队列和 guidance weight 时同时改变多个主变量。30 Hz 下 10 步约为 333 ms。

#### 第四步：调整 `max_guidance_weight`

- 过小：新旧动作块之间的一致性约束不足；
- 过大：可能过度遵循旧轨迹，对新观测响应迟缓，甚至表现为拉回、回弹或不稳定；
- Guided RTC 可从 **10.0** 开始实验。

本轮从初始值 **10.0** 下调为 **5.0**。这一调整发生在延迟和队列 bug 修复、`queue_threshold` 调为 20 之后，目的是单独减弱旧轨迹约束；`execution_horizon=10` 和 EXP schedule 保持不变。

### 2.3 最终配置的时间尺度

30 Hz 下一个控制步为 33.33 ms。实测稳态端到端推理约 147–159 ms，因此通常对应 5 个控制步：

```text
ceil(0.16 s × 30 Hz) = 5 steps
```

当前 `inference_delay=5`、`execution_horizon=10` 时，EXP schedule 的预期行为是：

- 前 5 步保持较强的旧动作约束；
- 后 5 步逐渐衰减；
- 第 10 步后不再使用旧动作前缀；
- `queue_threshold=20` 相比 `5 + 10 = 15` 保留 5 步余量，约 167 ms。

## 3. 延迟测量口径

延迟日志使用 CUDA 同步，避免只测到异步 kernel launch 时间。

| 指标 | 含义 |
|---|---|
| `model` | π0.5 `predict_action_chunk` 的同步模型推理时间 |
| `total` | 观测、预处理、模型推理和后处理的端到端时间 |
| `used_delay_steps` | 当前 RTC 实际采用的延迟步数 |
| `measured_delay_steps` | 当前端到端耗时换算出的控制步数 |
| `merge_delay_steps` | 合并动作块时实际跳过的动作数 |
| `actual_consumed_steps` | 推理期间 actor 真正执行的动作数 |

P50 是中位数；P95/P99 分别表示约 95%/99% 的样本不超过该延迟。RTC 参数以 P95 作为主要工作点，P99 用于监测更极端的尾延迟，而不是只参考平均值或模型 forward 时间。本文 P99 使用 nearest-rank 计算；当前样本量较小，因此它会接近或等于最大值，只作为尾部风险检查，不能替代更长时间的统计。

## 4. 调试过程

### 4.1 50 Hz：确认推理跨越约 8 个控制步

排除首次冷启动后，共记录 34 个 RTC guided chunks。

| 指标 | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Model latency | 140.33 ms | 139.50 ms | 146.35 ms | 167.38 ms | 167.38 ms |
| End-to-end latency | 144.45 ms | 143.57 ms | 150.81 ms | 171.64 ms | 171.64 ms |

50 Hz 的控制周期为 20 ms，稳态延迟通常换算为 8 步，偶发达到 9 步。首次冷启动为 406.32 ms，即 21 步。

同时发现 34 次稳态推理中有 12 次出现估算步数与实际消费步数相差 1 步，主要表现为估算 8 步、实际执行 7 步。

### 4.2 30 Hz、修复前：延迟变为 5 步，但错位仍存在

排除首次冷启动后，共记录 19 个 RTC guided chunks。

| 指标 | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Model latency | 142.98 ms | 141.65 ms | 152.32 ms | 156.19 ms | 156.19 ms |
| End-to-end latency | 147.13 ms | 145.79 ms | 156.84 ms | 160.40 ms | 160.40 ms |

降低控制频率没有明显改变模型的毫秒延迟，但将推理跨越的控制步数从约 8 步降到 5 步。19 次稳态推理的估算值均为 5 步，但实际消费为 4 或 5 步，其中 9 次相差 1 步。

这说明仅把控制频率从 50 Hz 改成 30 Hz，不能单独解决动作块边界的错位。

## 5. 根因定位

### 5.1 冷启动延迟永久污染 RTC

原实现将第一次 CUDA / 模型冷启动耗时写入 `LatencyTracker`，后续通过历史最大值计算 `inference_delay`。

在 50 Hz 下：

```text
first inference = 406 ms -> inference_delay = 21
```

在 30 Hz 下：

```text
first inference = 397–404 ms -> inference_delay = 12–13
```

当 `execution_horizon=10` 时，RTC 内部会将 prefix 起点截断到 10。对于 EXP schedule，这使前 10 步全部处于强引导区间，预期的渐变衰减消失。30 Hz 下 10 步对应约 333 ms，容易表现为动作迟滞、拉回或回弹。

### 5.2 `ceil(latency × fps)` 不等于实际消费步数

约 147 ms 的推理在 30 Hz 下会被向上取整为 5 步，但 actor 在真实线程调度中可能只执行 4 步。原实现无论实际执行数量是多少，都从新动作块丢弃 5 步，因此会偶发多丢一个动作，形成动作相位跳变。

### 5.3 队列索引与动作前缀不是原子快照

如果分别读取队列索引和剩余动作，actor 可能在两次读取之间继续执行一步，导致“起始索引”和“RTC 前缀”对应不同时间点。这也是潜在的一步错位来源。

## 6. 代码修复

本轮对 LeRobot RTC 真机评估脚本进行了以下修改：

1. 第一轮 `rtc_guided=False` 的冷启动仅记录，不写入稳态延迟跟踪器；
2. 第一个 guided chunk 使用 `rtc_bootstrap_latency_s=0.16`；
3. 后续使用 guided inference 的滑动 P95 估算 RTC 延迟；
4. 秒级延迟按当前 `fps` 自动换算，0.16 s 在 30/50 Hz 下分别得到 5/8 步；
5. 在 `ActionQueue` 锁内原子读取队列索引与 RTC 前缀；
6. 在同一把锁内计算推理期间实际消费的动作数，并使用该数值合并新动作块；
7. 增加同步延迟日志和 rollout 数据集 / 视频记录；
8. 增加相关单元测试，RTC 测试集共 150 项通过。

涉及的 LeRobot 文件：

```text
lerobot/examples/rtc/eval_with_real_robot.py
lerobot/src/lerobot/policies/rtc/action_queue.py
lerobot/src/lerobot/policies/rtc/latency_tracker.py
lerobot/tests/policies/rtc/test_action_queue.py
lerobot/tests/policies/rtc/test_latency_tracker.py
```

## 7. 30 Hz 修复后结果

修复后完成一次 60 秒真机 rollout，共记录 58 个 action chunks，其中 57 个为 RTC guided chunks。

| 指标 | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Model latency | 142.91 ms | 142.23 ms | 155.30 ms | 156.38 ms | 156.38 ms |
| End-to-end latency | 146.94 ms | 146.24 ms | 159.02 ms | 160.55 ms | 160.55 ms |

队列与 RTC 状态：

| 检查项 | 结果 |
|---|---:|
| `used_delay_steps=5` | 57 / 57 |
| `measured_delay_steps=5` | 57 / 57 |
| 实际消费 4 步 | 30 / 57 |
| 实际消费 5 步 | 27 / 57 |
| `merge_delay_steps == actual_consumed_steps` | 57 / 57 |
| ActionQueue mismatch warning | 0 |
| Queue underrun warning | 0 |
| Error / Traceback | 0 |

首次冷启动虽然为 404.46 ms、13 步，但日志显示：

```text
rtc_guided=False
used_delay_steps=0
measured_delay_steps=13
merge_delay_steps=0
```

因此冷启动没有进入 RTC guidance，也没有从首个动作块错误丢弃 13 步。

本轮还成功保存：

- 1,771 个数据集帧；
- 1,827 个已执行动作；
- Front / Side 两路 30 FPS rollout 视频；
- 完整同步延迟日志。

## 8. 首轮复现命令

以下命令使用 62k Full FT checkpoint 复现本轮工程调试。实际路径、CAN 口和摄像头设备需要按本机环境调整。

该命令依赖本文第 6 节描述的 LeRobot RTC 修改；未包含这些修改的上游版本可能没有 `rtc_bootstrap_latency_s`、`rtc_latency_percentile`、同步延迟日志和 RTC 数据集记录参数。

```bash
set -o pipefail

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_NAME="eval_pi05_full62k_rtc30_fix_${RUN_TIMESTAMP}"
LATENCY_LOG="outputs/rtc30_fix_latency_${RUN_TIMESTAMP}.log"
CHECKPOINT_PATH="outputs/train/pi05_full_finetune_test1102/checkpoints/062000/pretrained_model"
DATASET_ROOT="${HOME}/.cache/huggingface/lerobot/seeed_rebot_b601_rs/${EVAL_NAME}"

python -c '
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()

import runpy
runpy.run_path(
    "lerobot/examples/rtc/eval_with_real_robot.py",
    run_name="__main__",
)
' \
  --policy.path="${CHECKPOINT_PATH}" \
  --policy.device=cuda \
  --device=cuda \
  --rtc.enabled=true \
  --rtc.max_guidance_weight=5.0 \
  --rtc.execution_horizon=10 \
  --rtc.prefix_attention_schedule=EXP \
  --rtc_bootstrap_latency_s=0.16 \
  --rtc_latency_percentile=0.95 \
  --action_queue_size_to_get_new_actions=20 \
  --fps=30 \
  --duration=60 \
  --measure_inference_latency=true \
  --latency_log_every_n_chunks=1 \
  --robot.type=seeed_b601_rs_follower \
  --robot.port=can0 \
  --robot.id=follower1 \
  --robot.can_adapter=socketcan \
  --robot.cameras="{front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}}" \
  --task="slot battery" \
  --dataset.repo_id="seeed_rebot_b601_rs/${EVAL_NAME}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.fps=30 \
  --dataset.video=true \
  --dataset.num_image_writer_processes=0 \
  --dataset.num_image_writer_threads_per_camera=4 \
  2>&1 | tee "${LATENCY_LOG}"
```

> 最终正式评测应使用与训练数据一致的任务文本，并把 checkpoint 替换为需要评估的正式 checkpoint。本文保留 `slot battery` 是为了忠实记录本轮 62k 调试运行。

## 9. 日志验收标准

正常日志应类似：

```text
chunk=0 rtc_guided=False ... used_delay_steps=0 measured_delay_steps=13 merge_delay_steps=0
chunk=1 rtc_guided=True  ... used_delay_steps=5 measured_delay_steps=5 merge_delay_steps=4或5
chunk=2 rtc_guided=True  ... used_delay_steps=5 measured_delay_steps=5 merge_delay_steps=4或5
```

需要重点检查：

- guided chunks 的 `used_delay_steps` 是否稳定为 5；
- `merge_delay_steps` 是否始终等于 `actual_consumed_steps`；
- 是否出现 `Indexes diff is not equal to real delay`；
- 是否出现 queue underrun、Error 或 Traceback；
- rollout 结束后是否成功保存 episode 和两路视频。

## 10. 后续调参顺序

代码对齐问题修复后，再根据真机动作表现调整参数：

| 现象 | 首选调整 | 建议范围 |
|---|---|---|
| 仍有回弹或旧动作拉回感 | 降低 `max_guidance_weight` | 5.0 → 3.0 |
| 动作块交界仍偏硬 | 缩短 `execution_horizon` | 10 → 8，保持 EXP |
| 新动作衔接过松或突然转向 | 适度提高 guidance weight | 3.0 → 4.0/5.0 |
| 出现 queue underrun | 提高 queue threshold | 20 → 22/24 |
| 推理 P95 明显超过 166 ms | 重新测量 bootstrap / P95 | 5 → 6 steps |

每轮只调整一个主变量，并至少记录：checkpoint、任务文本、控制频率、RTC 参数、端到端延迟 P50/P95/P99、成功次数和失败阶段。

## 11. 当前结论与边界

- 将控制频率从 50 Hz 降至 30 Hz，不会显著降低模型的毫秒推理时间，但能把推理跨越的控制步数从约 8 步降至 5 步；
- 回弹不只由控制频率决定，冷启动污染和动作块错位是本轮定位到的关键工程问题；
- 修复后 57/57 个 guided chunks 实现了合并步数与实际消费步数一致；
- 当前日志证明延迟估计和队列对齐正常，但是否进一步提升任务成功率仍需按同一评测协议重新完成真机试验；
- 62k 调试结果不能直接替代仓库中 Full FT 80k + RTC 的 21/25 最终评测结果。

## References

- [LeRobot RTC implementation](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/rtc)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
- [Project π0.5 evaluation](./EVALUATION.md)
