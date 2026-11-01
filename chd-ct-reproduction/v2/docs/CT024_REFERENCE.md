# ImageCHD：多阶段分割与独立可解释诊断

0.24.0：训练与预测共用本折定位网络生成的裁剪框；分别保存原始投票、有限补全候选及最终分割，后处理只在本折验证集满足改善条件时采纳。加入显存预算、安全余量、运行中检查及失败记录，保持原 batch 和训练轮数；保护停止不等于显卡绝对不会 OOM。旧权重须另建任务重训，兼容预处理仍可复用。完整流程、选择条件、产物与限制见 [0.24.0说明](docs/roi-refinement-guard.md)。以下旧版本条目用于历史追溯，以新说明为准。

0.23.0：修正标签尺寸缩放的采样偏移，保持论文的 ROI 对齐和加权投票流程；重新核对原文后保留现有边界排除与区域生长，不采用直接保留边界的试改。模型与训练输入缓存启用新方法版本，旧模型需另建任务重训；原生/仅 Excel 物理重采样缓存可继续作为输入，旧额外尺寸缩放缓存须新建。历史结果继续只读查看。更新不会自动重跑训练，也不代表正式 Dice 已提高。论文未公开的实现细节及兼容要求见 [修复说明](docs/label-grid-boundaries.md)。

0.22.1：更新入口 `u` 分开计时，版本查询30秒、对象下载总上限900秒、代码备份180秒、应用更新120秒；显示实际 Git 进度和等待时间，保存分阶段日志与错误原因。下载失败不应用代码；应用阶段失败须核对实际 HEAD 与备份。旧版短超时的一次性升级办法见 [更新操作说明](docs/github-network.md)。

0.22.0：`c` 配置服务器代理与 SSH 密钥，`u` 更新代码，`w` 预览并上传已提交的 CT 代码，`v` 使用同一 SSH 密钥及 Git LFS 上传日志、指标和权重。无需 gh 或 API token；产物进入独立分支，不进入 main。旧版首次升级与操作见 [代理与私有仓库](docs/github-network.md)。

0.20.0：`o` 选择任务后进入简洁的结果中心，按分割、决策、诊断评估、训练与数据、文件与来源五个二级页查看。各页默认简洁，`d/s` 切换详细/简洁，`n/p` 翻页，`b` 返回上一级。命令行结果查询也默认简洁，显式选择详细可保留原全量输出。见 [结果面板](docs/result-details.md)。

0.19.0：结果页标明数据、真值、模型和预测来源，显示每折七结构 Dice、有效病例数、病例名单及逐病例指标；诊断部分展开每病训练可用性、实际决策路径、规则证据与逐疾病评估。旧结果无需重跑。操作和解释见 [详细结果说明](docs/result-details.md)。

0.18.0：新建任务可选择历史任务的预处理结果作为“输入任务”，直接沿用其最终缓存并跳过预处理。面板 `k` 可选择已完成任务和步骤，从头计算到独立的新任务目录；原权重、日志和结果保持不变。普通 `r` 仍是续跑/追加，不会自动重跑成功步骤。具体选择与命令见 [输入任务与重跑](docs/server-start.md#输入任务与主动重跑0180)。

0.13.0：测试和预测接入 GPU 并发上限；所有所选 GPU 工作完成后释放显卡，再恢复原网格、压缩和评估。非训练步骤内置 CPU 资源检查与有界病例并行，详细日志显示完成进度和估算剩余时间。训练已有的 CPU 调优、smoke/preflight 步骤与科学配置保持不变。改前文件见 [备份](back/before-inference-resources-20261019/manifest.json)，独立服务器小例子位于大 project 的 [server-resource-smoke](../server-resource-smoke/README.md)。


项目发布在 [tianwuasd/project 的 chd-ct-panel-v2 目录](https://github.com/tianwuasd/project/tree/main/chd-ct-panel-v2)。本地独立源码仓库通过项目发布目录同步，源码仓库本身可以没有 Git remote。

0.12.0 将三折 GPU 训练改为共享阶段队列：本折 `blood2d` 尚未结束时，空闲卡可提前训练后续折的独立模型。总并发受同一个 GPU 上限约束，每折 `blood_lstm` 仍只等待本折 `blood2d`；已完成折和完整 epoch 检查点可继续复用，CPU 训练保持逐折运行。改前源码见 [备份清单](back/before-cross-fold-queue-20261018/manifest.json)。

0.11.1 修复服务器面板拒绝 `gh-proxy.com` 镜像 origin 的问题，保留原地址并通过既有多镜像线路更新。旧面板的一次性升级方法见 [服务器说明](docs/server-start.md#更新程序与复用缓存)；修改前源码见 [备份清单](back/before-origin-alias-20261018/manifest.json)。

主目录默认流程：**预处理 → 三折训练（定位网络→预测框→其余网络→验证选择）→ 融合预测 → 独立评估**。每折仍有六个网络，另有定位框生成和验证后处理任务。数据使用 ImageCHD，主干按 CHD 论文图4明确结构实现、默认训练参数按表3设置；保留 BiConvLSTM、心脏定位及多尺度投票，血池辅助补全按本折验证结果决定是否采用。

这是原论文方法在现有七结构数据上的适配。当前没有初始血管标注、PV/SVC/IVC 标注与经验证的诊断规则，因此不运行两个 init 阶段。新增独立的解剖特征＋浅层决策树诊断，保留候选规则作对照；这是研究扩展，不宣称复现了论文临床准确率。

**核心实现与范围：** 见 [论文对齐记录](docs/paper-mapping.md) 和 [网络说明](docs/unet-source.md)。三维位置关联默认开启，正式配置使用论文规模，短测仍使用缩小网络。旧权重需重新训练，预处理复用遵守 [0.23.0 的采样兼容规则](docs/label-grid-boundaries.md)。修改前完整代码见 [备份说明](back/before-paper-alignment-20261013/SNAPSHOT.md)。

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
chd-ct-panel-v2/
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
├── src/chd_ct_v2/
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

### 历史结果直接查看（0.14.0）

服务器面板输入 `o`，按名称/编号选择项目任务，即可查看以前已完成任务的中文结果摘要。也可输入 `d` 指定原 workflow 路径。读取原有结果，不要求重跑或迁移；详情见 [结果查询](docs/server-start.md#历史结果查看0140)。

```bash
python3 start_server.py --query results --job-dir /原来的/任务_workflow
```

0.15.0 支持为已完成 workflow 追加未执行步骤（面板 `r` / `--append-tasks`），成功步骤校验后锁定复用，后续结果仍在面板 `o` 中显示。更新器首条有效线路成功即停止。新增面板 `v` / `--upload-job`，将日志、配置、指标及 best/last 模型权重打包上传；0.22.0 改用 SSH + Git LFS 独立产物分支，先预览再发布，失败重试原包。详见 [追加与上传](docs/results-publishing.md)。
