# AAOCA 原始 CT 到医生可编辑分割

TotalSegmentator 2.18.0 的三个预训练任务为原始 HU CT 生成十一种结构。0.6 命令行入口直接读取原图，保存 `aaoca_11.seg.nrrd`，成功导出核验后清理模型中间掩膜。CPU提前准备、GPU推理和CPU导出使用有界队列重叠执行。

## 模型和标签

| 编号 | 结构 | 任务 |
|---|---|---|
| 1、2 | 左心室、右心室 | heartchambers_highres |
| 3、4 | 左心房、右心房 | heartchambers_highres |
| 5 | 心肌 | heartchambers_highres |
| 6、7 | 主动脉、肺动脉 | heartchambers_highres |
| 8 | 肺静脉 | total |
| 9、10 | 上腔静脉、下腔静脉 | total |
| 11 | 冠脉整体 | coronary_arteries |

心腔名称与上游 heart_ 前缀一致。结构独立命名和着色，重叠区域保留。冠脉尚未细分分支，不自动判定开口异常或壁内段。

## 直接启动

在 AAOCA 根目录启动：

```bash
bash start_segmentation.sh install --runtime /srv/private/aaoca-runtime
bash start_segmentation.sh license --runtime /srv/private/aaoca-runtime
bash start_segmentation.sh check --runtime /srv/private/aaoca-runtime
```

入口直接接收参数。完整操作见[直接分割说明](../../SEGMENTATION_WORKBENCH.md)。心腔和冠脉任务的许可遵循上游要求；许可不写入源码、CSV 或命令行。

启动与自动安装使用 Python 3.11–3.13；已有 runtime 的独立环境和许可可以复用。

CSV 示例中的代号和路径均为虚构：

```csv
case_id,image,spacing_confirmed,hu_confirmed
sample_a,/srv/private/images/sample_a.nii.gz,true,true
```

确认字段必须有 HU 和真实毫米尺度依据，一个三维期相一行，同一影像不得重复。CSV 含私人路径，不公开。

```bash
bash start_segmentation.sh inventory --inventory /srv/private/research/CT总清单.json --data-root /srv/private/research --manifest /srv/private/input.csv
bash start_segmentation.sh plan --runtime /srv/private/aaoca-runtime --manifest /srv/private/input.csv --output-root /srv/private/labels --limit 1
bash start_segmentation.sh run --runtime /srv/private/aaoca-runtime --manifest /srv/private/input.csv --output-root /srv/private/labels --gpus 0,1 --workers 4 --io-workers 4 --prepare-workers 1 --export-workers 1 --prefetch 2 --threads 2 --min-free-gib 8 --gpu-reserve-gib 2 --background
```

示例要求 0、1 已分配给当前用户。每张卡同时处理一个病例，允许已有计算任务；空闲显存达到工作门槛加预留量时派发，默认共 10 GiB。每个模型开始前重新检查，实际并发还受 CPU/RAM/读写预算限制。显存门槛不构成硬限制或独占预约。--cpu 选择 CPU；--limit 和重复 --case-id 可限定体积。

模型内部完成必要重采样、裁剪与标准化，输出恢复输入网格。外部不保存重采样 CT、归一化 CT 或概率图副本。

workers覆盖CPU准备、GPU推理和CPU导出；默认卡数加准备并发1和导出并发1。准备中和已就绪病例受prefetch上限限制，原图及无损解压缓存均校验哈希。源数据或缓存损坏会报错。阶段开始/结束及就绪等待时间记录在私人events.jsonl中。

更新后，旧任务仍用原冻结代码。停止旧任务后，在新run参数追加 --continue-from 旧任务编号或latest，可核验并跳过已完成结果，让剩余病例使用新调度。旧结果原位保留，新任务记录结果引用，不改写旧任务。

## 医生编辑和保留文件

医生加载原始 CT 和结果根目录下的 `seg-<任务编号>/cases/<代号>/attempt-*/aaoca_11.seg.nrrd`，在 3D Slicer Segment Editor 中逐结构修订。文件包含名称、颜色和空间信息，重叠结构存于独立标签层。

成功后保留最终标签及小型私人来源、质检、恢复记录和日志。输入、代码/依赖、配置与输出哈希一致时复用标签；损坏时明确报错。中断推理重新执行该体积，导出未完成时保留掩膜。仅清理本任务已核验的掩膜，原始 CT 和外部历史成果保留。

医生需重点复核冠脉开口、主动脉根部、近端走行、肺动脉邻近边界和细小分支。空掩膜标为待复核，不据此认定结构不存在。

status、logs、stop、resume 使用同一 --runtime，默认选择最近的分割任务，也可用 --job 指定。任务请求与代码冻结，恢复沿用原任务。

## 验证与公开边界

格式验证使用两种小型几何样本和真实 pynrrd/slicerio 读写，核验重叠、斜切与负方向坐标、名称、颜色及错误拒绝。显式设置专用目录后运行：

```bash
AAOCA_SEG_TEST_ROOT=/path/to/approved-check-dir python -B -m unittest discover -s tests -p test_editable_format.py -v
```

模型的真实短测只证明其实际执行范围；目标服务器并发和临床质量分别验收。源码和通用说明可以公开；真实影像、病例编号、映射、清单、输出、许可、权重及私人日志不上传。

## 来源

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [Slicer 分割文件格式](https://slicer.readthedocs.io/en/latest/developer_guide/modules/segmentations.html)
- [设计](docs/label-only-design.md)与[实施计划](docs/label-only-plan.md)
