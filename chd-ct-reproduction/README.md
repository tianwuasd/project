# CHD CT：先天性心脏病 CT 诊断论文复现

对 Xu 等人在 **Medical Image Analysis 90 (2023), 102953** 的论文进行独立工程复现：
[A clinically applicable AI system for diagnosis of congenital heart diseases based on computed tomography images](https://doi.org/10.1016/j.media.2023.102953)。

**当前交付是可训练、可推理、可测试的研究实现，不是作者官方代码，也没有复现论文的临床准确率。** 未提供训练好的临床权重；缺少完整原始队列、作者决策规则和阈值。本仓库的诊断结果仅为算法研究候选，不用于患者诊疗。

## 实现范围

- 六个 3D U-Net 任务：两路 ROI、两路全心和两路初始血管；instance normalization、空间概率门控、加权 Dice + CE。
- 2D U-Net → 冻结 32 通道特征 → 两层双向 ConvLSTM；默认七切片序列，分别训练两个阶段。
- ROI 与多尺度融合，`1:1:2:2:2:2` 投票权重，初始七类到全心十一类的兼容投票，血池约束区域扩张。
- NIfTI 方向/affine 校验和原始空间输出；以患者为单位隔离训练、验证、测试。
- 连接、连通分量、体积、中心线图与物理半径特征；17 类疾病的三值规则接口。
- CPU 合成数据端到端示例、单元/集成测试、明确分母的评估工具。

**尚未完成的论文级复现**：作者精确网络参数量、血管主干与瓣膜定位、分支重分配、脊柱分割网络、完整诊断决策树和阈值、1282 例开发集的三折训练、2468 例临床测试及人机比较。近似实现逐项记录在 [论文映射](docs/paper-mapping.md)。

## 目录

```text
chd-ct-reproduction/
├── configs/                 # paper、CPU smoke 和候选规则
├── src/chd_ct/
│   ├── data.py              # NIfTI、manifest、ROI、标签
│   ├── models/              # U-Net / BiConvLSTM
│   ├── training/            # 分阶段数据集、增强、损失、训练
│   ├── inference/           # 多模型推理、融合、区域扩张
│   ├── features/            # 解剖特征、中心线图
│   ├── diagnosis/           # 可解释三值规则
│   ├── evaluation.py        # 分割与多标签评估
│   ├── synthetic.py         # 仅用于软件测试的几何体
│   └── cli.py               # 统一命令入口
├── tests/                   # 数值、几何、训练推理测试
├── scripts/                 # 一键合成示例
└── docs/                    # 数据规范、复现差异、设计、验证结果
```

`data/`、`runs/`、`checkpoints/`、影像和权重均被 Git 忽略。仓库不包含用户提供的 PDF 或患者数据。

## 安装

Python 3.11+；建议新环境。PyTorch 的 CPU/CUDA 版本请按 [官方安装说明](https://pytorch.org/get-started/locally/) 选择。

```powershell
cd D:\code_project\chd-ct-reproduction
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Linux/macOS 对应使用 `.venv/bin/python`。如本机要求代理，给 pip 命令添加 `--proxy http://127.0.0.1:7897`；受限目录出现缓存问题时添加 `--no-cache-dir`，并将 TEMP/TMP 指向可写目录。

本次执行使用隔离虚拟环境，并复用本机已有的 CPU PyTorch。具体版本与结果见 [验证记录](docs/verification.md)。

## 先跑通 CPU 示例

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py --output runs\demo
.\.venv\Scripts\python.exe -m pytest -q
```

示例依次生成 4 个无临床含义的几何体、校验数据、以缩小网络训练全部八个阶段、推理测试体并输出 JSON 与 NIfTI。每个阶段只运行两步；**这些输出不能用于解释精度或临床疾病**。脚本拒绝覆盖已有示例目录，重新运行时换一个输出路径。

输出位于 `runs/demo/prediction/`：`segmentation.nii.gz`、`initial_segmentation.nii.gz`、`features.json`、`diagnosis.json`、`inference.json`。可用 3D Slicer 打开影像与分割；本次不包含专用三维查看器。

## 接入真实数据

先阅读 [数据规范](docs/data-guide.md)，完成脱敏、类别映射和患者划分。数据集需要**全心标签**和单独的**初始血管标签**；普通 MM-WHS 标签并不等于本文标签。

```powershell
# 数据自检
chd-ct validate-data --manifest data/manifest.csv

# 正式配置仅建议在充足显存的 GPU 上使用
chd-ct train --config configs/paper.yaml --manifest data/manifest.csv --stage all --output checkpoints/fold0 --device cuda

# 一个完整模型组；三折训练完成后可在 --models 后依次传入 fold0 fold1 fold2
chd-ct infer --image data/test_case.nii.gz --models checkpoints/fold0 --output runs/test_case --device cuda --rules configs/rules.yaml

# 也可以跳过神经网络，对已有分割单独研究特征
chd-ct features --segmentation data/test_seg.nii.gz --initial data/test_initial.nii.gz --output runs/features.json
chd-ct diagnose --features runs/features.json --rules configs/rules.yaml --output runs/diagnosis.json
```

没有 `initial_label` 时可单独训练 `crop64/crop128/all64/all128/blood2d/blood_lstm`；完整七路推理需要两路 init 权重。`blood_lstm` 在同一输出目录读取 `blood2d.pt`，并将冻结编码器写入自身 checkpoint。

## 结果解释

规则输出为 `positive / negative / indeterminate`，每条结果附输入特征与比较条件。缺失结构、未公开阈值和未实现的精确解剖定位产生 `indeterminate`；同一互斥疾病组多条阳性也转为 `indeterminate`。`positive` 只表示示例规则满足，不等价于临床确诊。

`configs/rules.yaml` 是**自行构造的研究候选规则**。其中阈值为 `null` 的项必须由合法开发集和领域专家确定；不可使用测试集调参。额外确认特征可写入特征 JSON 进行方法实验，需自行记录标注来源。

分割评估：

```powershell
chd-ct evaluate-seg --prediction runs/test_case/segmentation.nii.gz --target data/test_seg.nii.gz --output runs/dice.json
chd-ct evaluate-diagnosis --truth data/truth.json --prediction runs/prediction_matrix.json --output runs/metrics.json
```

多标签矩阵按 `labels.py` 中 17 类的顺序排列，真值为 0/1，预测用 -1 表示无法判断。评估报告覆盖率、已决策准确率及所有槽位正确比例；不能直接对比论文按九个疾病组统计的 86.03%。

## 后续优先事项

1. 向作者申请 3750 例队列、1282 例开发集的标注/划分、初始血管规范、代码与阈值。
2. 用合法脱敏数据验证标签、方向和高分辨率边界，再逐个训练分割阶段。
3. 补齐 AO/PA 主干、瓣膜与分支定位，获得专家确认的诊断规则。
4. 固定患者级独立测试集，开展分割、诊断、消融和跨设备评估。

引用见 [CITATION.cff](CITATION.cff)。
