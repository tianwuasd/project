# ImageCHD：预处理、训练和预测分开运行

这是 ImageCHD 原生七结构分割适配，复用项目的 3D U-Net。它不调用原论文十结构融合、初始血管任务或疾病诊断规则。

## 三个独立步骤

| 步骤 | 本机脚本 | 服务器脚本 | 输入 → 输出 |
|---|---|---|---|
| 预处理 | `preprocess_imagechd.py` | `preprocess_server.sh` | 已解压 NIfTI → 可复用的 NPZ 缓存、划分和几何记录 |
| 训练 | `train_imagechd.py` | `train_server.sh` | 预处理目录 → 模型权重、训练历史 |
| 预测 | `predict_imagechd.py` | `predict_server.sh` | 预处理目录 + 权重 → 原始网格分割 NIfTI |

每一步都需要单独启动。训练和预测不会自动调用预处理；预测不读取分割真值、不修改划分、不更新模型。预处理与模型结果放在不同目录，已有输出会拒绝覆盖。每个脚本可用 `--help` 查看参数。

## 本机先跑通真实数据

以下命令在项目根目录、已安装项目依赖的 Python 环境中运行；Windows 可将 `python` 替换为 `.venv\Scripts\python.exe`。

```bash
# 1. 仅预处理；小样本测试选前4例完整标注，缺标注会列入排除报告
python preprocess_imagechd.py --input data/imagechd_raw/ImageCHD_dataset --output runs/my_imagechd/prepared --size 16 --limit 4

# 2. 仅短训练：一轮、一个训练batch和一个验证batch
python train_imagechd.py --prepared runs/my_imagechd/prepared --output runs/my_imagechd/model --config configs/imagechd7-smoke.yaml --mode smoke --device cpu

# 3. 独立预测；软件测试权重需要显式许可
python predict_imagechd.py --prepared runs/my_imagechd/prepared --checkpoint runs/my_imagechd/model/best.pt --output runs/my_imagechd/predictions --split test --allow-smoke --device cpu
```

当前本机原始数据已解压到 `data/imagechd_raw/ImageCHD_dataset`；`archive` 保留原始分卷。完整 96³ 预处理已保存在 `data/imagechd7_96`，可直接作为 `--prepared` 输入：97 例保留，13 例因缺少 MYO 标注排除，train/val/test 为 67/15/15。新脚本接收已解压目录，不会隐式下载解压器、复制压缩包或改写原始文件。

## 完整数据准备

```bash
python preprocess_imagechd.py --input /数据/ImageCHD_dataset --output /预处理/imagechd7_96 --size 96 --seed 42
```

默认保留全部七个前景结构都出现的病例，按固定种子划分约 70%/15%/15% 的 train/val/test。默认用文件名编号作为患者编号，这只是此公开数据集的工程假设；无法据此排除同一患者不同扫描。正式研究应核实患者身份并提供明确划分：

```csv
case_id,patient_id,split
ct_1001,patient001,train
ct_1002,patient002,val
ct_1004,patient004,test
```

CSV 需覆盖所有保留病例；同一患者不能跨集合。用 `--split-file /路径/split.csv` 载入。预处理后的清单固定保存，训练不会重新随机划分，也不会用 test 集更新或挑选模型。诊断表不参与本阶段划分，不把空白诊断补成阴性。本划分不声称病种分层或论文三折复现。

`--limit` 仅用于软件测试，带 limit 标记的预处理结果不能进入正式 train 模式。

预处理目录内：

- `dataset.json`：仅成功完成时生成，包含缓存相对路径、划分、来源影像 SHA256、强度参数和原始/标准方向几何。
- `preprocess-report.json`：成功/失败状态、排除原因；失败时保留现场，需用新的目录重试。
- `ct_*.npz`：归一化影像和可选的标签。迁移整个目录即可；训练/预测不依赖原 Windows 路径。

## 标签与强度规则

保持作者原生编号：0 背景，1 LV，2 RV，3 LA，4 RA，5 MYO，6 AO，7 PA。其余非负整数标签转换成 255，在交叉熵与 Dice 两部分损失中都忽略。负值/小数标签直接报错。任一目标结构未出现时，整例排除并列入报告；这是一种保守标注筛选，会改变样本构成，不能把被排除病例说成没有对应解剖结构。

每幅影像按自身第 1、99 百分位截断并归一化到 [0,1]，逐例保存上下限。不猜测 HU 偏移，不将原始 spacing=1 当作已验证毫米尺度。全体积转换为 RAS 轴序，再缩放到固定体素网格；没有用真值裁剪 ROI。此处不做按毫米间距的重采样，尺寸缩放可能改变长宽比例，是该简化基线的限制。

预测将低分辨率类别图用最近邻还原到原始尺寸/方向和提供的 affine；不输出毫米半径、疾病概率或临床诊断。低分辨率、少样本短测预测仅验证接口和计算流程。

## 训练模式

默认 `configs/imagechd7.yaml`：96³ 输入、base=16、4 层、batch=1、100 轮、Adam 学习率 0.0002、2 个计算线程。这是独立基线，不是论文网络超参数。

- `check`：只检查 train/val 缓存；不训练。
- `smoke`：保持缓存尺寸，改用 base=2、3 层、batch=1，只运行一个训练/验证 batch。
- `preflight`：保持指定模型结构与 batch，跑一轮、一个训练/验证 batch。
- `train`：执行指定完整配置；单独 Python 脚本不会隐式加跑预检，建议先运行 preflight。

模型配置中的 size 必须匹配预处理目录。修改输入尺寸需要新建预处理版本。服务器 train 模式自动先执行独立输出目录中的 preflight，通过后才从头开始正式训练；其他脚本职责不变。模型按验证损失保存 best.pt，不读取 test 标签；当前没有优化器续训接口。

## 无标签新影像的预测

先独立预处理，使用与权重一致的输入尺寸：

```bash
python preprocess_imagechd.py --input /影像/new_case.nii.gz --output /预处理/new_case --for-prediction --size 96
python predict_imagechd.py --prepared /预处理/new_case --checkpoint /模型/best.pt --output /预测/new_case --device cuda
```

`--for-prediction` 不要求标签，不生成训练划分，也不能用于训练。预测从缓存只读取 image 数组，即使训练时的标签文件已不存在仍可运行。正式 train 权重不需要 `--allow-smoke`。预测只接受 ImageCHD 七结构权重，拒绝原十结构论文模型、错误尺寸或不同归一化协议。

## zmic44 上使用

配置文件为 `configs/servers/zmic44.json`，依据用户指定的工作笔记中 **2026-09-26 更新**的记录。详见 [服务器说明](server-start.md)。

```bash
# 预处理独立执行，不占 GPU；下方原始数据位置需换成你实际上传/解压的位置
bash preprocess_server.sh --dataset /data_nas/zhangruichen/baidu_import/ImageCHD_dataset --non-interactive

# 复用刚生成的预处理结果，先短测
bash train_server.sh --mode smoke --non-interactive

# 正式训练：自动先做真实尺寸预检
bash train_server.sh --mode train --non-interactive

# 从训练日志找到 best.pt 后独立预测
bash predict_server.sh --checkpoint /完整路径/training/best.pt --split test --non-interactive
```

默认预处理目录为 `/data_nas/zhangruichen/chd_ct_data/imagechd7`。预处理执行一次即可；重复训练/预测不会重做。使用其他目录时，每个脚本传相同的 `--prepared /目录`。新扫描预测可用 `preprocess_server.sh --for-prediction --dataset /影像.nii.gz --prepared /新的预处理目录`，然后单独调用预测脚本。

三个服务器脚本都支持 `--runtime`、`--results`、`--python`、`--server-config`；训练/预测支持 `--gpu`。无图形界面的终端会提示输入路径，也可用上述参数完整指定。每一步先检查环境，再执行对应功能。报告留在 NAS 下的独立任务目录中。脚本前台运行，长任务请使用 tmux。
