# ImageCHD：多阶段分割与独立可解释诊断

项目发布在 [tianwuasd/project 的 chd-ct-reproduction 目录](https://github.com/tianwuasd/project/tree/main/chd-ct-reproduction)。本地独立源码仓库通过项目发布目录同步，源码仓库本身可以没有 Git remote。

0.12.0 将三折 GPU 训练改为共享阶段队列：本折 `blood2d` 尚未结束时，空闲卡可提前训练后续折的独立模型。总并发受同一个 GPU 上限约束，每折 `blood_lstm` 仍只等待本折 `blood2d`；已完成折和完整 epoch 检查点可继续复用，CPU 训练保持逐折运行。改前源码见 [备份清单](back/before-cross-fold-queue-20261018/manifest.json)。

0.11.1 修复服务器面板拒绝 `gh-proxy.com` 镜像 origin 的问题，保留原地址并通过既有多镜像线路更新。旧面板的一次性升级方法见 [服务器说明](docs/server-start.md#更新程序与复用缓存)；修改前源码见 [备份清单](back/before-origin-alias-20261018/manifest.json)。

主目录默认流程：**预处理 → 三折、每折六阶段训练 → 融合预测 → 独立评估**。数据使用当前 archive 中的 ImageCHD，U-Net 主干按 CHD 论文图 4 的层数、通道、InstanceNorm、LeakyReLU 和位置关联分支实现，默认训练参数按表 3 设置，并保留 BiConvLSTM、心脏区域定位、多尺度投票与血池边界细化流程。

这是原论文方法在现有七结构数据上的适配。当前没有初始血管标注、PV/SVC/IVC 标注与经验证的诊断规则，因此不运行两个 init 阶段。新增独立的解剖特征＋浅层决策树诊断，保留候选规则作对照；这是研究扩展，不宣称复现了论文临床准确率。

**核心实现与范围：** 见 [论文对齐记录](docs/paper-mapping.md) 和 [网络说明](docs/unet-source.md)。三维位置关联默认开启，正式配置使用论文规模，短测仍使用缩小网络。旧权重需要重新训练，Excel 重采样和已有预处理缓存继续保留。修改前完整代码见 [备份说明](back/before-paper-alignment-20261013/SNAPSHOT.md)。

三折按患者划分为约 56.7% 训练、10% 验证、33.3% 测试，每位开发集患者只进入一次测试折。开发集预测使用对应的留出折；独立新病例使用每阶段三个模型的概率平均。训练目录保存 `folds.json` 和 `training-summary.json`，服务器可查询各折 epoch/loss。三折修改前的 134 个文件另存于 [备份](back/before-threefold-training-20261015/SNAPSHOT.md)。

默认正式配置现已包含 CPU 数据加载优化：整例解压、三维血池标签和固定尺寸输入先做可复用缓存；已有小样本测试自动按 CPU 配额、内存、共享内存和并发数比较少量加载配置，后续训练复用推荐值。无需单独测试模式。推荐依据见 `cpu-tuning.json`，真实训练耗时见各阶段 `*.timing.json`。具体设置和验证边界见 [自动配置说明](docs/server-start.md#cpu-自动配置与小样本测试)。

## 从哪里开始

1. 先看 [使用流程](docs/workflow.md)，理解每一步的输入和输出。
2. 再看 [代码阅读路线](docs/code-guide.md)，依次读预处理、训练、预测。
3. 诊断模块从 [独立诊断流程](docs/diagnosis.md) 开始；空白疾病单元格按约定默认为阴性，并记录来源。
4. 准备上服务器时看 [服务器说明](docs/server-start.md)。

Windows 双击 **start.bat**，选择一个功能。数据路径可直接粘贴，或输入 `browse` 打开目录选择窗口。Linux/macOS 运行 `python start.py`。每个任务先检查环境；训练默认只做短测，正式训练需明确选择 `--mode train`。

| 功能 | 独立入口 | 服务器入口 | 输入 → 输出 |
|---|---|---|---|
| 预处理 | `preprocess.py` | `preprocess_server.sh` | 已解压 CT/标签 → 缓存、划分、空间记录 |
| 训练/短测 | `train.py` | `train_server.sh` | 缓存 → 三折各六阶段权重、models.json、训练摘要 |
| 预测 | `predict.py` | `predict_server.sh` | 缓存影像 + 模型目录 → NIfTI 分割 |
| 测试（预测 + 评估） | `test.py` | `test_server.sh` | 测试集缓存 + 模型 → 分割及 Dice |
| 评估 | `evaluate.py` | `evaluate_server.sh` | 预测 + 缓存真值 → Dice 报告 |
| 诊断各步骤 | `diagnosis.py labels/features/train/predict/evaluate/demo` | `diagnosis_server.sh --task ...` | 标签导入、特征、浅树训练、判断解释、诊断评估分别启动 |

服务器运行 `bash start_server.sh` 可多选步骤，默认独立后台运行，断开 SSH 后继续；`bash start_server.sh --status` 查看进度。菜单只询问所选步骤需要的设置，CPU 预处理/评估/诊断无需配置显卡。默认数据路径和 GPU 数在 `configs/servers/zmic44.json`，也可在菜单覆盖；`smoke_server.sh` 是独立短测。多卡采用一张卡一个阶段，保留血池 LSTM 的编码器依赖。显卡选择前显示实时状态；`bash start_server.sh --resume-job` 恢复最近中断，`--stop` 请求停止。详见 [服务器分步启动](docs/server-start.md) 和 [中断恢复](docs/recovery.md)。

训练和预测不会重新预处理。分割预测不会读取真值或训练模型；分割评估与诊断各步骤分别启动。

0.8.0 起，服务器选择的 GPU 数量是**最多并发数**。没有空闲卡时显示 `waiting_gpu` 并等待，有一张即可开跑，后来空闲的候选卡自动加入剩余阶段；长期只有一张可用也能串行完成。使用 `--status` 查看实际卡数，`--stop` 中断等待，`--resume-job` 恢复。候选卡始终受用户指定范围和 `CUDA_VISIBLE_DEVICES` 限制。

0.9.0 起，`bash start_server.sh` 先显示任务面板：运行中可查询进度、显卡、模型日志或停止；空闲时可新建并命名，或选择历史任务续跑。新建后仍可多选步骤。直接查询用 `--query models` 查看 epoch/loss，`--query logs --follow` 跟随模型详细日志；Ctrl+C 只退出查看。重名实验保存在不同目录，续跑保留原名称和配置。

0.10.0 起，面板提供 `x` 中断当前任务（保留检查点）、`u` 多镜像更新、`q` 退出；各输入页支持 `b` 返回和 `q` 退出。更新前确认目标任务并等待退出，备份代码，只做 Git 快进更新。预处理发现已有可识别缓存时自动进入校验复用/接续流程，来源或参数不兼容则拒绝覆盖。详见 [更新与缓存处理](docs/server-start.md#更新程序与复用缓存)。修改前文件保存在 `back/before-panel-update-20261016/`。

0.11.0 起，选择预处理时可以选“尝试沿用（失败自动新建）”或“新建”。缓存不兼容、损坏或无法识别时保留旧目录，自动生成新路径，记录原因并统一传给后续训练/预测。每次执行最多自动换目录重试一次；原始数据或系统错误正常报错。含预处理的流程先执行 CPU 环境检查和预处理，GPU 不足不会阻塞它们，实际 GPU 步骤才等待显卡。修改前代码见 `back/before-cache-selection-20261018/`。

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
python preprocess.py --input data/imagechd_raw/ImageCHD_dataset --output data/imagechd7_1mm --target-spacing 1 1 1
python train.py --prepared data/imagechd7_1mm --output runs/chd_smoke/models --mode smoke --device cpu
python predict.py --prepared data/imagechd7_1mm --models runs/chd_smoke/models --output runs/chd_smoke/predictions --split auto --allow-smoke --device cpu
python evaluate.py --prepared data/imagechd7_1mm --predictions runs/chd_smoke/predictions --output runs/chd_smoke/evaluation --split auto
```

独立入口的输出目录默认须为新目录；服务器预处理可校验并复用已有缓存。正式训练命令、已有缓存复用与无标签预测见 [使用流程](docs/workflow.md)。

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
│   ├── chd.yaml                  # CHD 论文规模，默认三折
│   ├── chd-smoke.yaml            # 16³ 单折软件短测/演示
│   ├── chd-cv-smoke.yaml         # 16³ 三折、18 阶段软件短测
│   ├── chd-author-unet.yaml      # 历史 Freiburg 配置，仅供备份版本使用
│   └── servers/zmic44.json       # 数据路径、GPU 数、环境、线程
├── src/chd_ct/
│   ├── imagechd/                 # 数据、六阶段训练、融合预测、评估
│   ├── diagnosis/                # 标签、解剖特征、规则、可解释树、诊断评估
│   ├── models/                   # CHD 论文主干 / 历史参考 / BiConvLSTM
│   ├── quickstart/               # 桌面引导与环境检查
│   └── server/                   # 环境安装、按需设置、多选流程、后台状态
├── tests/                        # 当前主流程测试
├── docs/                         # 当前使用方式和复现边界
└── back/paper-v1/                # 原版本的独立源代码快照
```

## 旧路径与复现范围

旧论文流程和之前的单 U-Net 适配均保存在 [back/paper-v1](back/paper-v1/ARCHIVE.md)，包括当时的代码、配置、文档、测试。主程序不导入 back，默认测试也不收集它；保留它是为了查阅和比较。

原论文：[A clinically applicable AI system for diagnosis of congenital heart diseases based on computed tomography images](https://doi.org/10.1016/j.media.2023.102953)。数据集、网络规模、标签、交叉验证与诊断部分的差异见 [论文对应关系](docs/paper-mapping.md)。

本地的 `archive/`、`data/`、`runs/`、影像、缓存、权重都由 Git 忽略。执行结果与未验证范围见 [验证记录](docs/verification.md)。
