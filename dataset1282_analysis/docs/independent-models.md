# Dataset1282 独立模型

STU-Net-S 对 CT 做自动多类别分割。SAM-Med3D Turbo 对 CT 和人工三维提示点做逐结构分割。各自运行、保存结果和评分。标签编号与既有 Dataset1282 字典相同。

## 环境和权重

先用 `scripts/setup_gpu.py` 检查研究 Python 与 CUDA 查询依赖。沿用训练预处理记录对应的 nnU-Net 2.8.1 和数值依赖。SAM-Med3D 单独安装 MedIM 0.1.2，可将其及 appdirs 1.4.4 安装到独立的依赖目录并加入 `PYTHONPATH`。本地权重直接读取，不需要联网加载模型。

- STU-Net-S 使用作者的 `small_ep4k.model`。加载编码器和解码器，重新初始化 11 通道输出头。[权重来源](https://huggingface.co/ziyanhuang/STU-Net/blob/75d046881d0273899b65bf3ef1ee601e9973e7c8/small_ep4k.model)
- SAM-Med3D 使用 `sam_med3d_turbo.pth`。固定发布版本并核对 SHA256 后严格加载全部张量。[权重来源](https://huggingface.co/blueyo0/SAM-Med3D/blob/07f6f6961f76ce5b30718472352ca09500ca8e53/sam_med3d_turbo.pth)

GPU 工作进程按 UUID 隔离，物理 0、1 号与其他获准设备一同参与选择。外部计算进程占用的卡跳过。资源不足时释放工作进程，等待重新准入。STU-Net 从最近完整轮继续，预测从已提交病例之后继续。运行期间查看各自的 `status.json`、`logs/` 和 `resources/`。

## STU-Net-S

`D` 指向已完成的 HU-marginal 预处理目录，`STU` 指向作者权重，`T` 和 `P` 是独立的训练与预测输出目录，`CFG` 和 `POLICY` 沿用创建预处理时的私有配置与政策。

```bash
"$PY" scripts/stunet.py train --prepared "$D" --weights "$STU" --output "$T"
"$PY" scripts/stunet.py predict --prepared "$D" --training "$T" \
  --config "$CFG" --policy "$POLICY" --output "$P"
"$PY" scripts/stunet.py evaluate --output "$P"
```

训练沿用冻结患者划分和部分标注损失。训练块默认 64×96×96，批量为 1，250 轮，每轮 250 次更新和 50 次开发取样。初始学习率为 0.001，使用 SGD 与深监督；以开发集固定取样的边际损失选择权重。CUDA 默认 BF16，CPU 默认 FP32，可用 `--precision fp32` 明确指定全精度训练。新机器应按实际容量选批量与训练块。

恢复训练或预测时使用相同输入和输出并加 `--resume`。训练可增加 `--epochs` 延长目标轮数。完整轮的模型、优化器和开发集历史落盘后才提交；非有限损失或梯度会终止并报告。推理使用 50% 重叠的高斯加权滑窗，输出背景与 10 个结构，掩膜恢复到原 CT 网格。

## SAM-Med3D Turbo

`SAM` 指向官方权重，`POINTS` 指向人工提示 JSON，`S` 是预测输出目录。

```bash
"$PY" scripts/sam_med3d.py predict --prepared "$D" --weights "$SAM" \
  --prompts "$POINTS" --config "$CFG" --policy "$POLICY" --output "$S"
"$PY" scripts/sam_med3d.py evaluate --output "$S"
```

提示文件按研究病例编号和结构编号登记。每个结构提供一个正点，坐标使用原 NIfTI 数组的零起始 XYZ 体素索引。

```json
{"example_case": {"1": [100, 120, 63], "6": [140, 128, 71]}}
```

推理读取原 CT，从 Excel 毫米网格直接采样以提示点为中心的 128³ 局部块，间距为 1.5 mm，各轴一致。正灰度体素提供 Z-score 统计量。每个结构输出独立的二值 NIfTI，以最近邻恢复到原网格，并记录预测是否接触局部块边界。局部块的裁剪位置和提示点均由用户提供的坐标确定。真实标签在评分阶段单独读取。[官方推理预处理](https://github.com/uni-medical/SAM-Med3D/blob/main/utils/infer_utils.py#L220-L244)

## 本地短测和正式评估

两套预测默认读取三个冻结留出组。可用 `--groups` 或 `--case` 选择子集。技术验收可使用 `--technical-subset --development` 读取已预处理的开发病例。训练短测可显式降低 `--epochs`、`--train-steps` 和 `--development-steps`。

正式评分记录各结构 Dice、HD95、ASD 和有效样本数。SAM 报告另记录缺提示、缺标和空预测。比较结果时按同一批病例和结构区分自动分割与单正点提示分割。

## SAM-Med3D 双卡训练短测

```bash
"$PY" -B -u scripts/sam_med3d_train.py run \
  --prepared "$D" --weights "$SAM" --gpus 2 --steps 2 --wait-seconds 30 --output "$TRIAL"
```

短测从训练组选择四个已标注左心室的独立病例，每卡每步处理一例。按推理同样的 1.5 mm 间距和 128³ 局部块产生训练输入，正点从对应训练掩膜内部取样。冻结影像编码器，使用 NCCL 同步更新提示编码器和掩膜解码器。调度进程为每个训练进程分配不同物理 UUID，每个进程只看到自己的显卡。默认从当前授权范围选择两张符合准入条件的空闲卡。

训练使用 FP32、AdamW、学习率 0.00008、权重衰减 0.1、梯度范数上限 1.0，以及二元交叉熵与软 Dice 之和。双卡全局批量为 2。实际参数写入 `request.json` 和 `result.json`。

启动前资源不足时，默认每 30 秒复查显卡和主机资源。等待期间 `status.json` 显示 `waiting_resources`，两张显卡同时通过准入后才创建训练进程。`--wait-seconds 0` 立即报告资源不足。

查看 `logs/sam-train-rank*.log`。两个进程都完成更新、参数摘要一致且权重写入 `last.pt` 后，才会生成 `result.json` 并标记 `passed`。
