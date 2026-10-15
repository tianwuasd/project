# 主流程使用说明

所有命令在项目根目录、已安装依赖的环境中执行。Windows 可把 `python` 换成 `.venv\Scripts\python.exe`。根目录的四个 Python 文件各自负责一项任务；需要中文引导和环境检查时使用 `start.py --task 对应功能`。

## 1. 独立预处理

```bash
python preprocess.py --input data/imagechd_raw/ImageCHD_dataset --output data/imagechd7_1mm --target-spacing 1 1 1
```

默认自动读取影像同目录的 `imagechd_dataset_image_info.xlsx`：按 `idx + 1000` 对应病例，采用两个 `PixelSpacing` 值和 `calculate_z_thick` 作为原始 X/Y/Z 间距，统一方向后重采样到 **1×1×1 mm**。CT 使用线性插值，标签使用最近邻插值；包括 `ct_1081` 在内的全部病例使用相同 Excel 规则。每幅影像仍按自身第1/99百分位截断并归一化到 [0,1]，不改变强度单位处理。

可用 `--spacing-excel /路径/imagechd_dataset_image_info.xlsx` 指定表格，或用 `--target-spacing 0.5 0.5 0.5` 改变目标间距。Excel 缺少病例、重复编号或间距无效时会报错，不会用 `(1,1,1)` 偷换源间距。没有同目录 Excel 且未显式指定时，保留原网格；`--no-resample` 可明确恢复旧行为。这些参数也可用于 `start.py --task preprocess` 和服务器入口。

输出含 `dataset.json`、`preprocess-report.json` 和逐例 NPZ。重采样缓存使用 `imagechd7-v3`，记录 Excel 哈希、逐例源间距、目标间距和往返网格变换；预测使用该变换恢复原始 NIfTI 尺寸、方向和头信息。原始文件不修改。Excel 或目标间距变化时必须使用新缓存目录，旧模型与缓存不会被覆盖。训练模型的各阶段仍自行缩放到配置中的张量尺寸；这次改动只增加预处理重采样，不改变网络输入策略。

默认排除缺少任一七结构标注的病例，其余额外整数标签记为255。排除表示标注不满足训练条件，不表示患者缺少该结构。预测用新影像可以没有标签，见最后一节。

小规模检查可加 `--size 16 --limit 4`；`--size` 大于0会在重采样后将整个体积缩放为立方体，改变比例，因此正式重采样缓存使用默认 `--size 0`。带 limit 标记的缓存不能用于正式训练。

## 已有缓存可以继续用

本机已经准备的 `data/imagechd7_96` 可直接作为 `--prepared`，无需重复处理：97例保留、13例缺少 MYO 标注排除，默认划分67/15/15。程序兼容旧 `imagechd7-v1` 格式，并记录 `cache_resolution=resized`。

96³缓存已丢失原始细节，把它送入128³/256²模型不会恢复这些细节。采用 Excel 重采样时另建 `data/imagechd7_1mm`，后续命令的 `--prepared` 指向新目录。旧预处理任务续跑可加 `--no-resample`；服务器缓存默认保存在 NAS。

## 2. 训练、短测和预检

```bash
# 六个阶段各一个训练batch和一个验证batch，模型缩小至16网格
python train.py --prepared data/imagechd7_96 --output runs/chd_check/models --mode smoke --device cpu

# 保持配置中的尺寸、深度、通道和batch，只跑一轮、每集合一个batch
python train.py --prepared data/imagechd7_1mm --output runs/chd_preflight/models --mode preflight --device cuda

# 正式运行全部六阶段
python train.py --prepared data/imagechd7_1mm --output runs/chd_full/models --mode train --device cuda
```

`check` 模式只读取 train/val 缓存检查有效性，不训练；test 不参与检查、更新参数或选模型。

六个阶段按顺序运行：

1. `crop64`、`crop128`：两个尺度的全图七结构分割，预测时用于确定心脏区域。
2. `all64`、`all128`：两个尺度的区域分割，训练区域来自训练真值，预测区域来自前两个模型。
3. `blood2d`：逐切片学习血池内部/边界/背景；血池由四心腔、AO、PA组成，MYO不属于血池。
4. `blood_lstm`：冻结 blood2d 特征编码器，复用双向 ConvLSTM 和七切片序列。

旧主干权重不能复用，需要重新训练；现有缓存无需重做。每阶段按验证损失保存最佳权重。`models.json` 只有六阶段完成后才标记 complete，记录配置、划分哈希和逐权重哈希。循环模型包含其实际使用的冻结编码器，防止模型混用。0.7.0 起另存每阶段 `.last.pt`（模型、优化器、随机状态、历史、最佳权重原始字节），可从完整 epoch 接续；旧版仅有最佳权重不能精确续训。见 [中断恢复](recovery.md)。

`configs/chd.yaml` 使用按原作者定义移植的有效卷积主干，3D 目标网格64³/128³、base16/4层，血池目标网格256²、base8/5层。网格适配会镜像扩展输入：实际3D输入为156³/220³，2D为444²。`configs/chd-author-unet.yaml` 使用作者主干宽度（3D base32、2D base64），类别数仍适配CHD；四个三维阶段默认接 gate，可分别用 `spatial_gate: false` 关闭。须先做 GPU preflight，尚未证实3090显存足够。来源、通道和开关详见 [U-Net说明](unet-source.md)。

独立 `train.py` 不隐式执行预检；桌面引导和服务器脚本的正式 train 模式会先运行 preflight，通过后才从头正式训练，两者输出分开。

## 3. 独立预测

```bash
python predict.py --prepared data/imagechd7_1mm --models runs/chd_full/models --output runs/chd_full/predictions --split test --device cuda
```

短测或预检权重必须显式加 `--allow-smoke`。`--models` 接收模型目录或其中的 models.json，旧单网络 best.pt 不能作为六阶段模型使用；其使用方法留在 back。

预测只读取缓存的 image，不读取 target。先预测区域，再进行四路1:1:2:2投票，随后用循环血池模型约束区域扩张。最后恢复原始尺寸、方向和 affine，保存 `*_seg.nii.gz`。同时保存缓存网格下的 `*_grid.npz`，供独立评估使用。报告为 `prediction-report.json`。

`--case-id ct_1001` 可选一例；不传 `--split/--case-id` 时预测全部输入病例。该命令只输出分割；疾病判断需要另外运行 [独立诊断步骤](diagnosis.md)。

## 4. 独立评估

```bash
python evaluate.py --prepared data/imagechd7_1mm --predictions runs/chd_full/predictions --output runs/chd_full/evaluation --split test
```

所选集合必须已全部完成预测。评估在缓存网格上进行：标签255被排除，预测和真值都缺失的类别为 null，不计入均值；缺失真值但出现假阳性的类别仍计入。输出逐例逐结构 Dice、逐例前景均值及其病例平均，保存在 evaluation.json。

重采样缓存的 Dice 在指定目标间距的缓存网格上计算；native 缓存对应原生方向变换后的网格；旧96³缓存对应低分辨率网格。比较实验时须使用相同网格。评估仍不输出毫米 HD95/ASD，也不把这个 Dice 报告称为临床诊断准确率。

## 固定患者划分

默认种子42按文件名编号划分约70%/15%/15%。文件名等同患者是工程假设；正式研究应核实患者身份并用 `--split-file` 指定 CSV：

```csv
case_id,patient_id,split
ct_1001,patient001,train
ct_1002,patient002,val
ct_1004,patient004,test
```

须覆盖全部保留病例，患者不能跨集合。预处理一次固定划分，后续训练不重新抽样。分割任务不使用疾病标签。独立诊断模块按用户约定将已有疾病列的空白作为阴性，并记录来源；见 [诊断说明](diagnosis.md)。

## 无标签新病例

```bash
python preprocess.py --input /影像/new_case.nii.gz --output /缓存/new_case --for-prediction
python predict.py --prepared /缓存/new_case --models /模型目录 --output /预测/new_case --device cuda
```

这是两个独立步骤。无标签缓存不能训练或评估；预测使用完整的已训练模型集合。
