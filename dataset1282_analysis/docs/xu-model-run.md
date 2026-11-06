# Xu 整套分割模型的运行与恢复

统一入口使用已完成的 0.3.0 `hu-marginal` 预处理目录。它在独立子进程中调用已有训练、取框、融合和评价命令。每次运行只对应一套监督组、权重和固定划分。整卷定位器及全结构分支使用正式模型配置。诊断及三折交叉验证留在后续阶段。

## 在服务器启动

进入本分支检出的 `dataset1282_analysis` 目录，选择已经分配给作业的 GPU 和装有 PyTorch、SciPy、nibabel、scikit-image、blosc2、psutil 的研究 Python。先确认该 Python 能识别目标 GPU，结果目录有足够空间保存检查点、辅助标签、分支预测和多份原网格分割。以下使用 Bash。

```bash
PY=/path/to/research/bin/python
D=/path/to/prepared/RUN/hu-marginal
O=/path/to/private/experiments/xu-marginal
GPU=cuda:0
test -s "$D/complete.json" || { echo '请等待预处理完成'; exit 1; }
"$PY" -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
mkdir -p "$(dirname "$O")"
args=(--prepared "$D" --output "$O" --supervision marginal --profile paper --device "$GPU" --threads 4)
# 如果有人工核准的起始段端面或内部界面，首次启动时加入并在恢复时沿用：
# args+=(--reviews /path/to/private/reviewed-planes.json)
nohup "$PY" -B -u scripts/xu_pipeline.py run "${args[@]}" >"${O}.launcher.log" 2>&1 < /dev/null &
echo "PID=$!"
```

边际监督组读取 590 个训练病例与 128 个开发病例，缺标按未知处理。完整标注对照使用单独输出目录，把 `--supervision` 改为 `complete`；该组训练和选权重所用的完整病例为 552 / 122。两组均报告固定开发集上可评价的结构，另列完全标注病例的评价。开发集参与选权重，结果作为开发集指标解释。

## CPU 与 GPU 资源策略

有两张已分配的完整物理 GPU 时，使用作业内可见的两个逻辑编号。当前按物理 UUID 核验，MIG 切片不在本版支持范围。64³ 分支用第一张，128³ 分支用第二张，定位、融合和血池也使用第二张。按 64³/128³ 成对并行训练、成对评价。只有两支都提交成功才进入依赖其结果的阶段。主分支的全局 batch 仍为 4，时序分支为 1，每支保留原 FP32、损失和 Adam 更新。当前最大同时运行两支。

先在原调度作业的 Bash 中用全新目录做双卡验收，无需准备患者数据。每张卡按正式规格各做一次更新，检查两张卡的身份和实际执行区间。随后使用相同卡序启动正式任务。目录名仅为示例。

```bash
"$PY" -B scripts/xu_pair_probe.py --devices cuda:0,cuda:1 --threads 4 --wait-seconds 60 --output "${O}-pair-probe"
args=(--prepared "$D" --output "$O" --supervision marginal --profile paper --devices cuda:0,cuda:1 --threads 4 --wait-seconds 60)
nohup "$PY" -B -u scripts/xu_pipeline.py run "${args[@]}" >"${O}.launcher.log" 2>&1 < /dev/null &
```

- 双卡采用 v2 资源策略。每卡预留总显存的 10% 且至少 4 GiB。程序在新阶段核对获准 GPU 的 UUID、空闲显存和外部计算进程，Linux 上必要证据缺失时拒绝启动。在实际工作进程内再检查并设置 PyTorch 分配器上限。容量短测使用正式形状；启动训练时再次比较当时余量、估算值和已测预留峰值。单卡默认沿用旧版显存预算；共享服务器单卡也可显式加 `--resource-policy v2`。保留作业调度器提供的 `CUDA_VISIBLE_DEVICES`。等待上限可设为 0–300 秒。
- CPU 数值线程与主机内存按当前系统/作业可用资源分配，两支并发时分摊预算。`--threads 4` 表示每支最多 4 个数值线程。可选 `--prefetch` 允许训练进程提前准备最多一个 CPU 批次；预算不足、来源大小未知时按需读取。`--prefetch-mib` 给出每支的附加内存上限，实际还受可用主存限制。每轮日志记录是否真正开启、CPU 等数据及传输调用耗时。
- 四路、六路及血池融合在同一运行内复用已核验的分支预测。数据、取框、权重、执行环境、代码身份须匹配。各融合目录保留各自结果及 `prediction-cache.json`，分别列出复用与新算数量。血池从二维编码器训练切换到时序网络时释放冻结编码器的残留梯度。
- `resources/` 保存阶段准入结果和拒绝原因，`status.json` 的 `active` 给出两个活动进程和各自日志。一个分支失败时，监督进程停止同组活动子进程；已提交的兄弟分支收据在恢复时核验复用。使用相同数据、代码、卡序和执行选项，以及 `--resume`。历史版本已启动的任务继续用原发布包恢复。
- 双卡物理重叠和正式队列每轮吞吐需在目标服务器分别实测。判断提速时比较相同输入、尺寸、批量和工作量的稳态耗时，同时核对取框和原网格预测。

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
