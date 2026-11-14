# Xu 整套分割模型的运行与恢复

统一入口使用已完成的 0.3.0 `hu-marginal` 预处理目录。它在独立子进程中调用已有训练、取框、融合和评价命令。每次运行只对应一套监督组、权重和固定划分。整卷定位器及全结构分支使用正式模型配置。诊断及三折交叉验证留在后续阶段。

## 在服务器启动

进入检出的 `dataset1282_analysis` 目录，先运行 `python3 scripts/setup_gpu.py --install --runtime "$R"` 并执行 `source "$R/training-env.sh"`。入口复用合格的研究 Python；GPU 查询所需的 `cuda-bindings` 缺失时，只在运行目录的 `gpu-query-py主次版本/` 中安装并导出本项目使用的 `PYTHONPATH`。以下命令沿用导出的 `PY` 和 `PYTHONPATH`。结果目录需有足够空间保存检查点、网络输入缓存和预测。

```bash
D=/path/to/prepared/RUN/hu-marginal
O=/path/to/private/experiments/xu-marginal
GPU=auto
test -s "$D/complete.json" || { echo '请等待预处理完成'; exit 1; }
"$PY" -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
mkdir -p "$(dirname "$O")"
args=(--prepared "$D" --output "$O" --supervision marginal --profile paper --devices "$GPU" --threads 4)
# 如果有人工核准的起始段端面或内部界面，首次启动时加入并在恢复时沿用：
# args+=(--reviews /path/to/private/reviewed-planes.json)
nohup "$PY" -B -u scripts/xu_pipeline.py run "${args[@]}" >"${O}.launcher.log" 2>&1 < /dev/null &
echo "PID=$!"
```

边际监督组读取 590 个训练病例与 128 个开发病例，缺标按未知处理。完整标注对照使用单独输出目录，把 `--supervision` 改为 `complete`；该组训练和选权重所用的完整病例为 552 / 122。两组均报告固定开发集上可评价的结构，另列完全标注病例的评价。开发集参与选权重，结果作为开发集指标解释。

## 自动使用作业获准的 GPU

正式服务器入口默认按作业内可见 GPU 建立资源池。可显式使用 `--devices auto`，或用 `--devices cuda:0,cuda:1,cuda:2` 指定分配到的逻辑卡。Linux 多卡自动模式沿用调度器的 `CUDA_VISIBLE_DEVICES` 或容器限定的可见设备。若调度器未设置可见范围，且整组物理 GPU 均获准，可用 `--devices all-visible` 自动枚举并筛掉忙卡。当前支持有可核验 UUID 的完整物理 GPU，MIG 实例须另行适配。

流水线与 GPU 资格检查默认 `--exclude-physical-gpus 0 1`，按 NVIDIA 物理编号及 UUID 排除，不受 CUDA 可见序号重新排序影响。其余可见卡统一查询后分配给互不重叠的并发分支。全部训练子进程先登记 PID、创建时间和父子关系，完成资源复查后统一放行。显式传空列表 `--exclude-physical-gpus` 可使用全部获准的可见设备。

**进程隔离。** 管理进程使用 NVIDIA 管理查询保存授权池，随后设置 `CUDA_VISIBLE_DEVICES=-1`。每个 GPU 工作进程在创建前得到仅含本组 UUID 的可见列表，局部卡号从 `cuda:0` 开始；启动参数、准入凭据和多卡成员同步转换。CPU 阶段同样设置 `-1`。工作进程复查外部占用后，核验实际 CUDA UUID 与顺序才开始计算。`resources/*-launch-*.json` 和 `*-isolation-*.json` 保留管理侧卡号、局部卡号、UUID 及实际核验结果。容量报告的卡号属于局部命名，父调度器按映射读取；`assignments/` 使用管理侧命名。

管理侧 `cuda:N` 按所继承授权列表的顺序命名，无 CUDA 掩码时按 NVIDIA 物理索引顺序命名；原 `CUDA_VISIBLE_DEVICES`、容器的 `NVIDIA_VISIBLE_DEVICES` 共同限制候选。CUDA 授权掩码使用 UUID；不能证明对应关系的数字 CUDA 掩码拒绝执行，避免异构卡排序和容器局部序号产生歧义。未知、重复或不明确的标识拒绝。GPU 训练与容量使用流水线或 fleet 入口，未隔离的直接 `xu_seg.py`、`xu_capacity.py`、`xu_accept.py` 及双卡工作进程调用拒绝执行。

```bash
# 先检查整组 GPU 是否能同时按病例预测，再核验正式 3D batch 的更新与显存。每次用新目录。
"$PY" -B scripts/xu_fleet_probe.py --devices auto --stage inference --threads 4 --output "${O}-check-inference"
"$PY" -B scripts/xu_fleet_probe.py --devices auto --stage 3d64 --threads 4 --output "${O}-check64"
"$PY" -B scripts/xu_fleet_probe.py --devices auto --stage 3d128 --threads 4 --output "${O}-check128"
# 作业仍在相同的 GPU 分配内时启动正式训练。
args=(--prepared "$D" --output "$O" --supervision marginal --profile paper --devices auto --threads 4 --wait-seconds 60)
nohup "$PY" -B -u scripts/xu_pipeline.py run "${args[@]}" >"${O}.launcher.log" 2>&1 < /dev/null &
```

| 工作 | 实际分配方法 |
|---|---|
| 两个独立的 3D 分支 | 64³、128³ 可同时运行。每个正式分支固定 batch 4，一次更新最多 4 张卡。6 张显存余量足够的卡可分成 3+3，各分支上的样本分布为 2+1+1。8 张可分成 4+4。短批次只派发实际存在的病例。可用卡太少、显存不对称或 CPU/RAM 不够同时训练两支时，分别评估能否顺序执行。 |
| 单分支多卡更新 | 每卡计算各自样本，按原顺序将结果汇总到主卡，使用原有的完整标签或缺标边际损失及原 Adam 更新。主卡须能承载优化器与汇总结果。 |
| 血池 | `blood` 固定选择一张显存余量足够的 GPU。二维编码器一次 4 张切片；时序输入保持一个病例的七张连续切片。两阶段在同一张卡完成，显存不足时释放并重新选一张合格卡。 |
| 定位、单支评价、四路/六路及血池融合的模型预测 | 冻结模型按病例轮转分配，逐卡驻留预测器。可用卡、待处理病例数与实时主机内存共同决定并发量。例如有 12 张合格卡和足够病例与主存时，这些阶段能使用 12 张。原网格 CPU 收尾、审核与血管图处理使用 CPU。 |

**调度和恢复。** 每个新阶段重新筛选资源。v2 预留每卡总显存的 10% 且至少 4 GiB，核对 UUID、实时剩余显存、阶段需求、外部计算进程及 CPU/cgroup 证据，实际子进程逐卡复核并设置显存分配器上限。外部计算进程的存在记录在资源报告中，准入由显存预算决定。活动分支共享一份 CPU/RAM 预约，已核实的兄弟进程私有内存可抵扣其尚未兑现的预留。健康阶段保持当前成员；确认 GPU 资源变化后，停止本任务的活动组并确认退出，重新选卡及做容量检查，从最新完整 epoch 恢复训练。设备数量可以变化，全局 batch、损失、学习率及轮数保持不变。未完成的融合归档到 `attempts/` 后重新计算，完成阶段继续验证原产物。

CPU 内存压力触发减少供数 worker、单批队列和分支顺序执行，调整保存到 `resources/recovery.json`，手动恢复也继续采用。资源不足时主流程保留锁，状态为 `waiting_resources`，工作进程已释放。等待间隔由 `--wait-seconds` 确定，至少 1 秒；超过 `--max-reallocations 3` 后逐步延长至最多 300 秒，资源恢复后自动续跑。此参数控制等待退避阈值。没有可用卡时沿用 v1 的等待方式；训练 batch 保持不变。数据、数值、身份及 GPU 隔离错误直接停止；资源查询的不可解析错误也停止。

`assignments/` 记录候选、分配、历史设备组及有效工作槽；`reallocations/` 保存变化原因、原分配和可恢复检查点；`resources/` 保存预算与运行证据。训练日志打印恢复的完整 epoch，`status.json` 的 `active` 包含进程、设备组及日志。显存余量足够的卡在下一阶段或重新分配时参与选择，blood 保持单卡。开发数据和缓存版本变化需用新目录。代码变化须匹配精确兼容审核或使用原包恢复。

**CPU 和预测复用。** 预取默认开启，可用 `--no-prefetch` 做实测对照。`--prefetch-workers 4` 是每支供数线程上限，`--prefetch-mib 16384` 是首选队列预算；真实病例单批估算更大时，在本阶段 RAM 份额内提高到该单批的需要值，分别记录 requested 与实际额度。启动前读取真实准备后尺寸；合成容量检查与真实供数预算分别核验。下一批无法加入队列时先交付已经就绪的批次，总预算计入当前批次，实时新增额度只保留未完成工作的估算。每次扩容重新读取 RAM/cgroup 余量，不等待 CPU 利用率采样。只有一个 CPU 工作槽时采用按需供数。

实际并发受 CPU 份额、数值线程、源病例尺寸及实时 RAM 限制。多个线程按原顺序交付完整批次，增强和随机数继续在训练线程执行。`--threads 4` 控制每支数值线程数。日志记录实际 worker、队列峰值、内存预约、等待与复制时间；并行 worker 的准备时间是累计值，不能与等待时间相加当作总耗时。CUDA 使用固定页内存、独立复制流和事件同步，主机输入保留至复制完成。

3D 训练目录的 `.input-cache/` 保存同一尺寸与框的 NumPy 输入，校验源文件、变换代码和内容摘要后复用。损坏会终止任务。每轮 `input_cache_cumulative` 记录累计命中与生成次数；它属于可重新生成的加速产物，不列入检查点完成凭据。三次融合继续复用签名一致的分支预测，命中数量在 `prediction-cache.json`。模型、batch、FP32、采样、损失及学习率保持既定设置。

**Blood 二维供数。** `--prefetch` 同时适用于 blood 的训练和验证，复用相同的有界队列与锁页传输。每批按二维 512×512 输入、七帧窗口及源病例解压工作区单独估算；不会用三维网络输入的体素公式估算切片。CPU worker 只读取和缩放图像、标签，增强仍在原消费线程按原随机序列执行。每例结束回收该例队列，CPU 槽不足时改为按需串行供数。不会通过改变 batch、精度或轮数缩短训练。

Blood 每例记录 `epoch`、`total_epochs`、`cases_completed`、`slices` 和 `timing`；其中 `cpu_wait_seconds` 是训练线程等待供数的时间，`cpu_prepare_seconds` 是所有 worker 准备时间之和，`step_seconds` 是主线程提交计算、更新和读取 loss 的耗时。`transfer_call_seconds` 为主机传输调用耗时，不能单独当成 GPU DMA 时间。每轮记录 `epoch_seconds`、`remaining_epochs`、`train_timing`、`validation_timing`。`phase_eta_seconds` 只估算当前阶段，病例日志指当前训练或验证遍历，轮末日志指当前 encoder 或 sequence 的剩余轮次，不包含后续模型、融合和资源等待。

**运行期间检查。** 管理进程通过 NVIDIA 管理接口查询全卡状态，CUDA Driver 枚举仅在已限制可见 UUID 的工作进程执行。外部计算 PID 记录在报告中，设备准入取决于实时显存预算。父进程在每次检查结束后等待 1 秒继续检查，将本阶段内存、主机/cgroup 余量、GPU 外部进程和显存余量写入 `resources/*-runtime.json`。同时核查本阶段及管理进程是否出现在未分配的 GPU，隔离违规直接停止并留证。资源不足触发重新分配或等待并保留最后完整 epoch，所有外部进程保持运行。调度检查无法阻止另一用户在两次检查之间启动任务，服务器仍需遵守共同资源分配规则。

可执行 `"$PY" scripts/xu_acceleration_check.py --prepared "$D" --output "$W/check-input-new"` 测量实际病例的串行、并行首次缓存、串行热读取和缓存复用。该检查采用小型 Conv3d 验证输入与 Adam 更新，计时只用于供数验收；完整模型吞吐通过正式试跑日志测量。

`scripts/xu_fleet_probe.py` 在一次资格检查中会分组覆盖可用卡。正式 3D 一批最多 4 例，各组各做一次真实更新；该诊断的总耗时与正式训练每轮吞吐分别测量。只有一张卡且已获准的本机可显式选 `--device cuda:0 --exclude-physical-gpus`（独立资格检查用 `--devices cuda:0 --exclude-physical-gpus`）。单卡 Linux 共享运行默认 v2，本机小显存兼容策略可显式传 `--resource-policy legacy`。

3D 容量检查继承该阶段的预取开关、内存预算和 worker 数，按合成输入的实际体素数量预留供数内存。`capacity.json` 的 `supply` 记录预取是否实际启用、队列峰值、内存预留和传输计时；正式模型的检查仍需完成一次有限损失与非零参数更新。

本地回归使用相同容量输入生成器、64³/128³、batch 4 和轻量 Xu 网络，实际比较串行与预取后的损失、权重及随机状态。正式宽度 CPU 检查仅在当前内存满足准入时执行；目标服务器 GPU 容量以其实际运行报告为准。

## 阶段与产物

| 阶段目录 | 工作与产物 |
|---|---|
| `crop64`、`crop128` | 正式两路整卷定位训练，保存最佳权重、恢复点和逐轮损失。也可显式提供同一实验的 `--crop64`、`--crop128` 两份已有定位权重，程序核对模型、输入及监督合同。 |
| `localize`、`audit-roi` | 从图像生成严格投票框，三轴总边长各扩 20%。冻结候选后才读取原网格标签审核。空框、整图、任一网格框占比超 95%、整结构漏截、分组逐结构覆盖率第 5 百分位低于 95% 均阻止下游训练。通过才写 `audit-roi/boxes.json`。 |
| `all64`、`all128`、`evaluate-all64`、`evaluate-all128`、`four-way` | 使用预测框训练两个全结构模型，分别评价，再与两个整卷分支融合并输出四路基础结果。 |
| `targets`、`init64`、`init128`、`six-way` | 根据经核准的端面构建近端目标。正式近端训练至少需要本监督组内 4 名合格训练患者和 1 名合格开发患者。满足条件后成对训练两分支并融合；不足时记录人数与原因，保留四路结果。 |
| `blood`、`with-blood` | 血池由九种组成结构推导。同一监督组的训练、开发各需至少 1 例合格病例。两阶段训练切片网络和时序网络，再将血池约束加到可用的四路或六路融合结果上。 |
| `vessels` | 在已完成的最完整融合结果上做血管图处理，逐例记录改标、保留或拒判原因。 `cases/*.nii.gz` 是开发病例的十结构分割，输出位于预处理包保留的原生局部毫米网格。 |

`result.json` 汇总受监督人数、缺少的辅助分支、各阶段逐类 Dice、HD95、ASD、有效分母和空预测数。`evaluation.json` 保留在各评价目录中。比较顺序是四路、六路、血池修正、血管后处理。可用监督不足允许有据可查地跳过辅助分支；数据及数值异常停止并记录，临时容量不足进入资源等待。完整实验的运行时间和精度需要在正式队列测量。

## 查看、分段和恢复

### 停止后更新

在启动训练时的工程目录中，以旧实验输出目录 `O` 为目标正常停止。命令验证主机、输出路径、进程创建身份和启动参数，等待父任务及登记进程退出；无关进程保持运行。可先读取远端停止脚本，停止后再更新本地源码：

```bash
set -o pipefail
git fetch origin main &&
git show origin/main:dataset1282_analysis/scripts/xu_stop.py | "$PY" - "$O" &&
git pull --ff-only origin main
```

停止失败时禁止继续启动。更新后使用原输出目录和原参数追加 `--resume`，日志使用 `>>` 追加保存。仅 `configs/xu-resume-compat.json` 审核的完整旧、新源码映射允许跨代码更新继续；本次包含 `c5ced51` 对应的隔离版本。原 `run.json`、已完成收据和检查点保持原身份，实际新执行源码另存于 `.runtime/resume-compat/`。输入、模型或未审核代码变化仍拒绝。未完成轮次重跑，已有完整轮次继续；历史未完成 blood 多卡分配会重新选择单卡。不要在运行中覆盖该实验使用的代码。

### 正式模型先试跑一至两轮

Linux 可用一个薄启动脚本串接输入检查、CPU 合成训练/恢复检查、三项 GPU 检查及正式两轮试跑。`RUN` 必须是尚不存在的新目录，父目录先创建；任一检查失败都不会启动训练。代码、预处理路径和环境在该次实验中保持不变。

```bash
RUN=/path/to/private/experiments/xu-trial-unique
mkdir -p "$(dirname "$RUN")"
nohup bash scripts/xu_trial.sh "$PY" "$D" "$RUN" "$GPU" 2 >"${RUN}.launcher.log" 2>&1 < /dev/null &
# 训练状态为 $RUN/training/status.json，启动参数保存在 $RUN/settings.sh。
```

先在目标 GPU 上依次运行上面的 `inference`、`3d64`、`3d128` 容量检查。检查使用合成输入；两个 3D 检查执行正式尺寸、batch 4 的一次 Adam 更新。全部通过后，在新的实验目录使用：

```bash
nohup "$PY" -B -u scripts/xu_pipeline.py run "${args[@]}" --pause-after-epoch 2 >"${O}.launcher.log" 2>&1 < /dev/null &
```

这会让 `crop64`、`crop128` 各完成两轮训练和开发集评价后返回 `paused`，不启动取框及后续分割。每轮仍遍历所选监督组的全部训练和开发病例；不是两个 batch。`--pause-after-epoch 1` 可先只跑一轮。模型尺寸、batch、原定 15 轮学习率计划及监督方式均不变；暂停只控制本次运行的停止位置。两支的 `last.pt` 和 `training.json` 保存已完成轮数，暂停阶段不会获得完成凭据。

从一轮继续到两轮，在原命令加 `--resume` 并将暂停点改为 `2`。决定继续正式训练时，移除暂停选项、保持其他参数与代码不变，并加 `--resume`；程序接着最近完整轮次训练至原定 15 轮，再执行取框审核。暂停选项仅用于自行训练正式定位器，不能搭配外部冻结定位权重。请保留该实验启动时的代码版本，运行中不要更新源码。

```bash
"$PY" scripts/xu_pipeline.py status --output "$O"
tail -f "${O}.launcher.log"
# 只跟踪当前阶段的逐轮损失或逐例处理进度：
LOG=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("log") or "")' "$O/status.json")
if test -n "$LOG"; then tail -f "$LOG"; fi
```

首次运行可加 `--stop-after four-way`，在四路评价和权重落盘后停下。继续时在相同配置后加 `--resume`，去掉 `--stop-after`。中断时训练从最近完整的 epoch 恢复；只支持全阶段重做的步骤会把原尝试移到本输出目录的 `attempts/`。每阶段的文件摘要存于 `receipts/`，恢复或重读完成结果时会核对这些文件。权重、监督组、研究环境或审核输入变化时请使用新目录；仅经过完整源码映射审核的实现变更可以保留原目录续跑。

```bash
nohup "$PY" -B -u scripts/xu_pipeline.py run "${args[@]}" --resume >>"${O}.launcher.log" 2>&1 < /dev/null &
```

状态为 `roi_rejected` 时查看 `audit-roi/localization_qc.json`。相同配置再次恢复只验证既有拒收证据。两个定位器得以改进后，使用新实验目录重新生成候选和审核。正式任务在训练前用所选设备做同尺寸、同精度的一次带优化器更新的容量检查，结果在 `capacity/`；血池另查编码器和七层序列。校验按照当前设备余量保守准入，调度器分配的 GPU 需在运行期间持续可用。技术用 `--profile smoke --technical-subset` 只接受技术子集，其结果单独保存。
