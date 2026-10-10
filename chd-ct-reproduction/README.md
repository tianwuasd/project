# ImageCHD：多阶段分割与独立可解释诊断

主目录现在只有一条默认流程：**预处理 → 六阶段训练 → 融合预测 → 独立评估**。数据使用当前 archive 中的 ImageCHD，U-Net 主干按原作者发布的 Caffe 网络定义移植，保留独立可关闭的 gate，并沿用 BiConvLSTM、心脏区域定位、多尺度投票与血池边界细化方法。

这是原论文方法在现有七结构数据上的适配。当前没有初始血管标注、PV/SVC/IVC 标注与经验证的诊断规则，因此不运行两个 init 阶段。新增独立的解剖特征＋浅层决策树诊断，保留候选规则作对照；这是研究扩展，不宣称复现了论文临床准确率。

**U-Net 来源与开关：** 见 [官方来源与移植说明](docs/unet-source.md)。三维 gate 默认开启。旧权重需要重新训练，预处理缓存可复用。

## 从哪里开始

1. 先看 [使用流程](docs/workflow.md)，理解每一步的输入和输出。
2. 再看 [代码阅读路线](docs/code-guide.md)，依次读预处理、训练、预测。
3. 诊断模块从 [独立诊断流程](docs/diagnosis.md) 开始；空白疾病单元格按约定默认为阴性，并记录来源。
4. 准备上服务器时看 [服务器说明](docs/server-start.md)。

Windows 双击 **start.bat**，选择一个功能。数据路径可直接粘贴，或输入 `browse` 打开目录选择窗口。Linux/macOS 运行 `python start.py`。每个任务先检查环境；训练默认只做短测，正式训练需明确选择 `--mode train`。

| 功能 | 独立入口 | 服务器入口 | 输入 → 输出 |
|---|---|---|---|
| 预处理 | `preprocess.py` | `preprocess_server.sh` | 已解压 CT/标签 → 缓存、划分、空间记录 |
| 训练/短测 | `train.py` | `train_server.sh` | 缓存 → 六阶段权重、models.json |
| 预测 | `predict.py` | `predict_server.sh` | 缓存影像 + 模型目录 → NIfTI 分割 |
| 测试（预测 + 评估） | `test.py` | `test_server.sh` | 测试集缓存 + 模型 → 分割及 Dice |
| 评估 | `evaluate.py` | `evaluate_server.sh` | 预测 + 缓存真值 → Dice 报告 |
| 诊断各步骤 | `diagnosis.py labels/features/train/predict/evaluate/demo` | `diagnosis_server.sh --task ...` | 标签导入、特征、浅树训练、判断解释、诊断评估分别启动 |

服务器数据路径和 GPU 数统一在 `configs/servers/zmic44.json` 设置，也可在菜单选择；`smoke_server.sh` 是独立短测。多卡采用一张卡一个阶段，保留血池 LSTM 的编码器依赖。详见 [服务器分步启动](docs/server-start.md)。

训练和预测不会重新预处理。分割预测不会读取真值或训练模型；分割评估与诊断各步骤分别启动。

诊断先试跑：`python start.py --task diagnosis-demo --device cpu --non-interactive`，随后查看输出中的 `diagnosis-demo/prediction/diagnosis-report.html`。

## 安装与先跑通

需要 Python 3.11+。在项目根目录的独立环境中运行：

```bash
python -m pip install -e ".[dev]"
python start.py --task environment --device cpu --non-interactive
python start.py --task demo --device cpu --non-interactive
```

Windows 已有环境可使用 `.venv\Scripts\python.exe`。合成演示会明确生成测试影像并串起六阶段训练、预测、评估，只验证软件流程。自己的数据按下面四个独立步骤运行：

```bash
python preprocess.py --input data/imagechd_raw/ImageCHD_dataset --output data/imagechd7_native
python train.py --prepared data/imagechd7_native --output runs/chd_smoke/models --mode smoke --device cpu
python predict.py --prepared data/imagechd7_native --models runs/chd_smoke/models --output runs/chd_smoke/predictions --split test --allow-smoke --device cpu
python evaluate.py --prepared data/imagechd7_native --predictions runs/chd_smoke/predictions --output runs/chd_smoke/evaluation --split test
```

所有输出目录须为新目录。正式训练命令、已有缓存复用与无标签预测见 [使用流程](docs/workflow.md)。

## 主目录结构

```text
chd-ct-reproduction/
├── start.bat / start.py           # 中文功能选择、环境检查
├── preprocess.py                 # 独立数据准备
├── train.py                      # 六阶段训练
├── predict.py                    # 融合推理
├── evaluate.py                   # 单独评估
├── diagnosis.py                  # 独立诊断六个子命令
├── *_server.sh / start_server.py # 同一主流程的服务器入口
├── configs/
│   ├── chd.yaml                  # 实用单卡配置
│   ├── chd-smoke.yaml            # 16³ 软件短测
│   ├── chd-author-unet.yaml      # 原作者主干宽度/深度，三维 gate 默认开启
│   └── servers/zmic44.json       # 数据路径、GPU 数、环境、线程
├── src/chd_ct/
│   ├── imagechd/                 # 数据、六阶段训练、融合预测、评估
│   ├── diagnosis/                # 标签、解剖特征、规则、可解释树、诊断评估
│   ├── models/                   # 原作者结构主干 / 网格适配 / gate / BiConvLSTM
│   ├── quickstart/               # 桌面引导与环境检查
│   └── server/                   # 环境安装、GPU检查、任务派发
├── tests/                        # 当前主流程测试
├── docs/                         # 当前使用方式和复现边界
└── back/paper-v1/                # 原版本的独立源代码快照
```

## 旧路径与复现范围

旧论文流程和之前的单 U-Net 适配均保存在 [back/paper-v1](back/paper-v1/ARCHIVE.md)，包括当时的代码、配置、文档、测试。主程序不导入 back，默认测试也不收集它；保留它是为了查阅和比较。

原论文：[A clinically applicable AI system for diagnosis of congenital heart diseases based on computed tomography images](https://doi.org/10.1016/j.media.2023.102953)。数据集、网络规模、标签、交叉验证与诊断部分的差异见 [论文对应关系](docs/paper-mapping.md)。

本地的 `archive/`、`data/`、`runs/`、影像、缓存、权重都由 Git 忽略。执行结果与未验证范围见 [验证记录](docs/verification.md)。
