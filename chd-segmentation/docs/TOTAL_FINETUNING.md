# ImageCHD fold0 的 TotalSegmentator 微调

## 初始化权重和环境

模板指定 `heartchambers_highres` 心腔模型 `Dataset301_heart_highres_1559subj`。`model_folder` 指向已有的 `nnUNetTrainer__nnUNetPlans__3d_fullres`，该目录应含 `plans.json`、`dataset.json` 和 `fold_0/checkpoint_final.pth`。 [官方心腔模型配置](https://github.com/wasserth/TotalSegmentator/blob/v2.18.0/totalsegmentator/map_tasks_config.py#L365-L376)

`license_home: auto` 按以下顺序选择第一个含 `config.json` 的目录。也可以在私有配置中直接指定路径。

1. `TOTALSEG_HOME_DIR`
2. `AAOCA_SEG_RUNTIME/private/totalseg`
3. `~/.local/state/aaoca-seg-terminal/private/totalseg`
4. `~/.totalsegmentator`

`status` 只显示配置路径及文件是否存在。`check` 和301运行入口会调用官方离线检查及在线许可验证；官方服务不可达或未认可许可时终止。离线检查只检查号码长度，在线检查向官方服务提交号码。许可证号码保留在原文件中，代码及日志不保存号码。微调和衍生权重的分发权限以实际许可条款为准。 [官方许可检查](https://github.com/wasserth/TotalSegmentator/blob/v2.18.0/totalsegmentator/config.py#L102-L149)

可手动指定已有的公开 `total` CT 模型进行独立实验，例如 `Dataset293_TotalSegmentator_part3_cardiac_1559subj`。训练流程读取所指定的本地权重，权重下载与许可证登记由部署人员通过官方入口完成。 [公开293权重](https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset293_TotalSegmentator_part3_cardiac_1559subj.zip)

Linux 服务器上使用独立的 `/data4/zhangruichen/env/totalsegmentator` 环境。依赖固定为 `TotalSegmentator==2.18.0`、`nnunetv2==2.8.1`，并使用该版本兼容的 CUDA PyTorch 和 NCCL。nnU-Net 2.8.1 支持 Python 3.10 及以上版本，依赖声明排除 PyTorch 2.9.x。项目本体和 PyYAML、psutil、filelock 等公共依赖也需在该环境可导入。环境和权重由部署人员按服务器现有配置准备。 [nnU-Net 依赖声明](https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/pyproject.toml#L1-L54)

已有独立环境时，从清华镜像安装本项目的 `total` 依赖组：

```bash
CODE=/data4/zhangruichen/code/chd-segmentation
PY=/data4/zhangruichen/env/totalsegmentator/bin/python
TMPDIR=/data4/zhangruichen/env/totalsegmentator "$PY" -m pip install --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple "${CODE}[total]"
"$PY" -m pip check
```

AAOCA 的默认已有权重目录为 `~/.local/state/aaoca-seg-terminal/weights`；设置了 `AAOCA_SEG_RUNTIME` 时使用该目录下的 `weights`。将实际存在的301三维模型目录写入私有配置的 `model_folder`。

## 准备、训练和推断

复制 `configs/finetuning/totalsegmentator.yaml` 到 course 下的私有配置目录，核对 `prepared` 和 `model_folder`。从仓库根目录运行以下命令。

```bash
mkdir -p /data4/zhangruichen/course/chd-segmentation/configs
cp -n configs/finetuning/totalsegmentator.yaml /data4/zhangruichen/course/chd-segmentation/configs/totalsegmentator.yaml
```

```bash
COURSE=/data4/zhangruichen/course/chd-segmentation
PY=/data4/zhangruichen/env/totalsegmentator/bin/python
CFG="$COURSE/configs/totalsegmentator.yaml"
export PYTHONPATH="$PWD/src"
"$PY" -B -m chd_segmentation.finetuning.total check --config "$CFG"
"$PY" -B -m chd_segmentation.finetuning.total prepare --config "$CFG"
"$PY" -B -m chd_segmentation.finetuning.total train --config "$CFG"
"$PY" -B -m chd_segmentation.finetuning.total status --config "$CFG"
"$PY" -B -m chd_segmentation.finetuning.total predict --config "$CFG" --role test --checkpoint best
```

- `prepare` 读取既有冻结的 ImageCHD fold0 来源描述，按完整原始 HU 影像和已核实病例间距导出 NIfTI，空间单位写为毫米。重复准备时核对逐病例的来源、处理规则和已导出文件摘要，复用已完整导出的病例。冻结的训练和验证名单写入 nnU-Net 的 `splits_final.json`，原测试病例单独保存影像输入。训练病例生成 `dataset_fingerprint.json`，CT 归一化继续使用源计划的统计。目标输出头按 ImageCHD 的八个类别初始化，预先检查其他所有参数及缓冲区的键和形状。训练与推断均采用完整病例和 nnU-Net 自身的预处理，保留主动脉与肺动脉的可用视野。
- `train` 默认等待两张空闲的 CUDA 卡并独占使用，全局批量4，每卡2个样本。默认200轮，每轮250次训练更新和50次验证迭代，初始学习率0.001。首次准备前可用 `epochs`、`batch`、`gpu_count`、`learning_rate` 调整训练，`gpu_ids` 可指定与卡数相同的设备索引或 UUID。`gpu_ids: null` 由资源管理器选择空闲设备。心肌未标注的病例继续使用既有部分标注损失，按有监督类别计算验证指标。
- 中断后在 `train` 命令加 `--resume`。恢复前核验冻结来源、初始化权重、预处理计划、训练记录及检查点。训练目录保留 `checkpoint_latest.pth`、`checkpoint_best.pth`、`checkpoint_final.pth` 各自适用的状态。
- `predict` 可选择验证集或测试集。使用微调后检查点的目标类别定义，输出对应 ImageCHD 类别编号及原始 NIfTI 几何。生成标准 `predictions.json`，可交给项目现有的 `evaluate` 命令，继续按有监督类别计算 Dice。完整病例预测按此命令单独启动。

## 输出位置

| 内容 | 位置 |
| --- | --- |
| 配置、训练记录、阶段日志及模型检查点 | `course/chd-segmentation` 的配置及运行目录 |
| 导出的原始 HU 影像、标签和 nnU-Net 预处理缓存 | `data/chd-segmentation/ImageCHD-TotalSegmentator-fold0` |
| 验证或测试预测及清单 | `result/chd-segmentation/ImageCHD-TotalSegmentator-fold0` |

正式预处理会新增完整 NIfTI 输入、预处理数据和验证标签副本。阶段资源预算可在配置中设置；读取病例头和源计划估算整例处理占用，取估算与手动预算的较大值，再增加50%余量。准备和训练保留100 GB项目上限；预测遵守整机80%限制。配置中的24 GiB对应至少36 GiB预留。官方整例准备或预测超出可容纳预算时直接终止，并输出原因。该入口使用官方 nnU-Net 整例预处理和预测，实际占用和每卡显存需在服务器核验。资源监控发现超额会终止子进程，训练可从上一轮完整检查点恢复。
