# STU-Net 服务器运行入口

代码位于 `/data4/zhangruichen/code/chd-segmentation`；环境使用 `/data4/zhangruichen/env/chd-segmentation`。已有输入、权重和冻结计划继续使用。更新、预处理、训练分别执行。

本次代码安装命令由对应发布记录给出完整 SHA，必须带 `--force`。常规更新入口为：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 -B /data4/zhangruichen/code/chd-segmentation/update.py --force --mirror direct
```

服务器通过现有网络直接访问 GitHub，单次更新清除继承的代理环境变量，Git 下载不使用代理配置。Windows 本机的联网设置独立维护。

从已有冻结来源描述重新启动 ImageCHD fold0 预处理，已经完整提交的病例会复用：

```bash
mkdir -p /data4/zhangruichen/course/chd-segmentation/logs
nohup bash /data4/zhangruichen/code/chd-segmentation/scripts/ImageCHD-CHD1282-STUNet-20261115_run.sh imagechd 0 preprocess >> /data4/zhangruichen/course/chd-segmentation/logs/ImageCHD-STUNet-20261115_fold0.log 2>&1 </dev/null &
```

对应来源描述为 `/data4/zhangruichen/course/chd-segmentation/prepared/ImageCHD-STUNet-20261115_fold0_source.json`，完整离线描述为同目录的 `ImageCHD-STUNet-20261115_fold0_offline.json`。正式数组在 `/data4/zhangruichen/data/chd-segmentation/preprocessed`。`preprocess` 阶段使用既有冻结计划；`prepare` 阶段用于首次准备来源描述。

本次 ImageCHD 已核实的完整配方为 `e402c48392467f20a726a8a5af18a91e9ca9262d5dcef66806b2529630051753`，数组目录为 `/data4/zhangruichen/data/chd-segmentation/preprocessed/ImageCHD-STUNet-20261115/fold_0/e402c48392467f20`。重启 `preprocess` 复用该目录内完整核验的病例，并补充未完成病例。来源描述内嵌冻结计划；配置指向的计划文件和原始清单须继续存在且摘要一致，缺失时直接报告路径，不重新拟合训练统计。GPU 数量、batch 和 epoch 不参与该离线配方身份。

两套数据现有输入约定如下；路径以既有服务器运行配置为准，CHD1282 的配方目录由其 `preprocessed/index.json` 中的身份记录定位：

| 数据集 | 配置 | 来源描述 | 完整离线描述 | 正式数组 |
| --- | --- | --- | --- | --- |
| ImageCHD | `course/chd-segmentation/inputs/imagechd_stunet_fold0.yaml` | `course/chd-segmentation/prepared/ImageCHD-STUNet-20261115_fold0_source.json` | `course/chd-segmentation/prepared/ImageCHD-STUNet-20261115_fold0_offline.json` | `data/chd-segmentation/preprocessed/ImageCHD-STUNet-20261115/fold_0/e402c48392467f20` |
| CHD1282 | `course/chd-segmentation/inputs/chd1282_stunet_fold0.yaml` | `course/chd-segmentation/prepared/CHD1282-STUNet-20261115_fold0_source.json` | `course/chd-segmentation/prepared/CHD1282-STUNet-20261115_fold0_offline.json` | `data/chd-segmentation/preprocessed/CHD1282-STUNet-20261115/fold_0/<冻结配方身份前16位>` |

表中相对位置均从 `/data4/zhangruichen` 开始。原始数据分别使用 `data/ImageCHD_dataset`、`data/dataset1282_CHD_CT_back`，权重继续使用 `data/pretrained/STU-Net/small_ep4k.model`。本轮无需再次上传清单、划分或权重。

CHD1282 fold0 的同类入口：

```bash
nohup bash /data4/zhangruichen/code/chd-segmentation/scripts/ImageCHD-CHD1282-STUNet-20261115_run.sh chd1282 0 preprocess >> /data4/zhangruichen/course/chd-segmentation/logs/CHD1282-STUNet-20261115_fold0.log 2>&1 </dev/null &
```

预处理完成后再启动 ImageCHD fold0，两张空闲 GPU、global batch 4、200 epoch：

```bash
nohup env CHD_GPUS=2 CHD_BATCH=4 CHD_EPOCHS=200 bash /data4/zhangruichen/code/chd-segmentation/scripts/ImageCHD-CHD1282-STUNet-20261115_run.sh imagechd 0 train >> /data4/zhangruichen/course/chd-segmentation/logs/ImageCHD-STUNet-20261115_fold0.log 2>&1 </dev/null &
```

已有完整检查点时将末尾阶段 `train` 改成 `resume`；配置、数据、模型和代码完整审计须相容。显卡默认不共享，不足两张时持续等待。需要指定物理显卡时在 `env` 中添加 `CHD_GPU_IDS=5,7` 等实际编号；实际编号须处于服务器可见范围。上述后台命令退出终端后继续运行，日志均在 course；预测与评价仍在 result。

```bash
tail -f /data4/zhangruichen/course/chd-segmentation/logs/ImageCHD-STUNet-20261115_fold0.log
```

最新内存快照及病例/块进度位于 `/data4/zhangruichen/course/chd-segmentation/runs/ImageCHD-STUNet-20261115_fold0/` 内的 `memory.json`、`progress-preprocess.json`、`progress-train.json`。块总量按当前阶段真实网格计算，未知总量明确显示未知。
