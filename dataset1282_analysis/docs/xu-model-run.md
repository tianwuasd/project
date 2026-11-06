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
