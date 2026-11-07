# Xu 整套分割模型的运行与恢复

统一入口使用已完成的 0.3.0 `hu-marginal` 预处理目录。它在独立子进程中调用已有训练、取框、融合和评价命令。每次运行只对应一套监督组、权重和固定划分。整卷定位器及全结构分支使用正式模型配置。诊断及三折交叉验证留在后续阶段。

## 在服务器启动

进入本分支检出的 `dataset1282_analysis` 目录，选择已经分配给作业的 GPU 和装有 PyTorch、SciPy、nibabel、scikit-image、blosc2、psutil 的研究 Python。先确认该 Python 能识别目标 GPU，结果目录有足够空间保存检查点、辅助标签、分支预测和多份原网格分割。以下使用 Bash。

```bash
PY=/path/to/research/bin/python
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
| 两个独立的 3D 分支 | 64³、128³ 可同时运行。每个正式分支固定 batch 4，一次更新最多 4 张卡。6 张相近且空闲的卡可分成 3+3，各分支上的样本分布为 2+1+1。8 张可分成 4+4。短批次只派发实际存在的病例。可用卡太少、显存不对称或 CPU/RAM 不够同时训练两支时，分别评估能否顺序执行。 |
| 单分支多卡更新 | 每卡计算各自样本，按原顺序将结果汇总到主卡，使用原有的完整标签或缺标边际损失及原 Adam 更新。主卡须能承载优化器与汇总结果。 |
| 血池 | 二维编码器一次 4 张切片，最多 4 张卡。时序输入保持一个病例的七张连续切片，冻结编码器可在最多 7 张卡上提取特征，随后按原顺序运行序列网络。 |
| 定位、单支评价、四路/六路及血池融合的模型预测 | 冻结模型按病例轮转分配，逐卡驻留预测器。可用卡、待处理病例数与实时主机内存共同决定并发量。例如有 12 张合格卡和足够病例与主存时，这些阶段能使用 12 张。原网格 CPU 收尾、审核与血管图处理使用 CPU。 |

**调度和恢复。** 每个新阶段重新筛选资源。v2 预留每卡总显存的 10% 且至少 4 GiB，核对 UUID、空闲显存、外部计算进程及 CPU/cgroup 证据，实际子进程逐卡复核并设置显存分配器上限。活动分支共享一份 CPU/RAM 预约，已核实的兄弟进程私有内存可抵扣其尚未兑现的预留。阶段内固定工作量与成员；中断后已开始的阶段使用已记录的同一组设备和最新完成的检查点。在下一阶段重新选卡。`assignments/` 记录每阶段的候选、分配、占用或未用的卡及有效工作槽，`resources/` 记录预算，训练日志和推理分片记录可核对逐卡实际工作量。`status.json` 的 `active` 包含进程、设备组及日志。开发数据和缓存版本的变化需用新目录。旧版冻结任务使用原包恢复。

**CPU 和预测复用。** `--threads 4` 为每个活动分支的数值线程上限。显式加 `--prefetch` 可提前读取最多一批，`--prefetch-mib` 给出每支附加内存上限。供数日志记录实际启用、主机等待及传输耗时。三次融合可复用签名一致的分支预测；签名包含数据、取框、权重、环境及实际设备分片，各自的 `prediction-cache.json` 列出命中和新算数量。血池阶段切换会释放冻结参数残留梯度。

`scripts/xu_fleet_probe.py` 在一次资格检查中会分组覆盖可用卡。正式 3D 一批最多 4 例，各组各做一次真实更新；该诊断的总耗时与正式训练每轮吞吐分别测量。只有一张卡的本机可显式选 `--device cuda:0`（独立容量命令用 `--devices cuda:0`）。单卡 Linux 共享运行默认 v2，本机小显存兼容策略可显式传 `--resource-policy legacy`。

## 阶段与产物

| 阶段目录 | 工作与产物 |
|---|---|
| `crop64`、`crop128` | 正式两路整卷定位训练，保存最佳权重、恢复点和逐轮损失。也可显式提供同一实验的 `--crop64`、`--crop128` 两份已有定位权重，程序核对模型、输入及监督合同。 |
| `localize`、`audit-roi` | 从图像生成严格投票框，三轴总边长各扩 20%。冻结候选后才读取原网格标签审核。空框、整图、任一网格框占比超 95%、整结构漏截、分组逐结构覆盖率第 5 百分位低于 95% 均阻止下游训练。通过才写 `audit-roi/boxes.json`。 |
| `all64`、`all128`、`evaluate-all64`、`evaluate-all128`、`four-way` | 使用预测框训练两个全结构模型，分别评价，再与两个整卷分支融合并输出四路基础结果。 |
| `targets`、`init64`、`init128`、`six-way` | 根据经核准的端面构建近端目标。正式近端训练至少需要本监督组内 4 名合格训练患者和 1 名合格开发患者。满足条件后成对训练两分支并融合；不足时记录人数与原因，保留四路结果。 |
| `blood`、`with-blood` | 血池由九种组成结构推导。同一监督组的训练、开发各需至少 1 例合格病例。两阶段训练切片网络和时序网络，再将血池约束加到可用的四路或六路融合结果上。 |
| `vessels` | 在已完成的最完整融合结果上做血管图处理，逐例记录改标、保留或拒判原因。 `cases/*.nii.gz` 是开发病例的十结构分割，输出位于预处理包保留的原生局部毫米网格。 |

`result.json` 汇总受监督人数、缺少的辅助分支、各阶段逐类 Dice、HD95、ASD、有效分母和空预测数。`evaluation.json` 保留在各评价目录中。比较顺序是四路、六路、血池修正、血管后处理。可用监督不足允许有据可查地跳过辅助分支；训练异常、数据损坏或容量不足则停止并记录。完整实验的运行时间和精度需要在正式队列测量。

## 查看、分段和恢复

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

首次运行可加 `--stop-after four-way`，在四路评价和权重落盘后停下。继续时在相同配置后加 `--resume`，去掉 `--stop-after`。中断时训练从最近完整的 epoch 恢复；只支持全阶段重做的步骤会把原尝试移到本输出目录的 `attempts/`。每阶段的文件摘要存于 `receipts/`，恢复或重读完成结果时会核对这些文件。权重、代码实现、监督组、研究环境或审核输入变化时请使用新目录。

```bash
nohup "$PY" -B -u scripts/xu_pipeline.py run "${args[@]}" --resume >>"${O}.launcher.log" 2>&1 < /dev/null &
```

状态为 `roi_rejected` 时查看 `audit-roi/localization_qc.json`。相同配置再次恢复只验证既有拒收证据。两个定位器得以改进后，使用新实验目录重新生成候选和审核。正式任务在训练前用所选设备做同尺寸、同精度的一次带优化器更新的容量检查，结果在 `capacity/`；血池另查编码器和七层序列。校验按照当前设备余量保守准入，调度器分配的 GPU 需在运行期间持续可用。技术用 `--profile smoke --technical-subset` 只接受技术子集，其结果单独保存。
