# STU-Net 资源与划分设置

配置写在 `configs/experiments/*.yaml`。命令行同名参数优先生效；`run.json` 保存完整生效配置，恢复时核对数据、计划、训练参数、代码和依赖身份。

## 显卡和 batch

`resources.device: cuda` 启用 CUDA；`gpu_count` 支持 1–4。两份正式配置默认请求两卡，全局 `training.batch: 4`。自动分担分别为 `[4]`、`[2,2]`、`[2,1,1]`、`[1,1,1,1]`。

训练命令可增加以下参数：

| 设置 | 参数示例 |
|---|---|
| 自动挑选两张卡 | `--device cuda --gpus 2 --batch 4` |
| 指定物理卡 | `--device cuda --gpu-ids 0,2 --batch 4` |
| 指定三卡及份额 | `--device cuda --gpu-ids 0,1,3 --batch 4 --batch-layout 2,1,1` |
| 修改训练轮数 | `--epochs 200` |
| 自动供数 | `--workers auto` |
| 限制全组供数进程 | `--workers 8` |
| 持续等待显卡 | `--wait-seconds forever` |
| 指定等待时限 | `--wait-seconds 600` |
| 关闭繁忙卡共享 | `--no-share-gpu`（默认） |
| 手动开启繁忙卡共享 | `--share-gpu` |
| 允许在一至两卡间调整 | `--gpus 2 --min-gpus 1` |
| 恢复同一任务 | `--resume` |

`gpu_ids` 中的整数使用 `nvidia-smi` 的物理编号，也支持完整 GPU UUID；继承的 `CUDA_VISIBLE_DEVICES` 限制仍有效。启动后核对实际 UUID。epoch 内的检查点恢复要求相同进程数量；完整 epoch 边界允许在配置范围内调整，保留完整训练状态。

默认关闭共享，只使用空闲卡；数量不足时持续等待。显式设置 `--share-gpu` 才允许满足条件的繁忙卡参与组合。每张繁忙卡分别核对同设备、同实现、同环境、同规格、每卡 batch 和精度的显存记录，以及默认 30 秒观察窗口、利用率 P90 ≤50% 和显存余量。手动 `gpu_ids` 保持指定顺序。

默认 `resources.wait_seconds: null`，持续等待直到凑齐所需显卡；等待期间不登记部分显卡占用。`--wait-seconds forever` 可以覆盖旧配置中的有限等待值，整数或小数秒数可以设置等待时限，`0` 只检查一次。Ctrl+C 可以停止等待。配置、数据或驱动错误仍直接报错，CPU/内存准入保护保持生效。

轮询默认每 2 秒查询一次，等待详情按日志心跳间隔输出，避免长时间等待产生过多日志。事件记录继承的 `CUDA_VISIBLE_DEVICES`、实际可见物理编号、手动选择、各卡剩余显存、对应训练份额的拒绝原因和等待时间。GPU 数量不足、被本项目预留、缺少峰值记录、观察窗口不足、利用率高及显存不足分别说明。

```yaml
resources:
  share: false
  min_gpus: null
  wait_seconds: null
  poll_seconds: 2
```

缺少实测记录时该繁忙卡暂不参与组合；显存不足时保留用户设置的 batch 和 patch。

显存记录使用 Torch 峰值保留量，加上实际进程显存中 allocator 以外的占用。完成训练及验证的有界配置可以建立对应规格记录；技术小块记录不能用于正式大块准入。检查点保存模型、优化器、scaler、逐进程 RNG、全局抽样游标与轮次/验证进度，文件独立读回通过后才替换 latest，随后清理旧的检查点。

200 epoch 默认每轮 250 次训练尝试、50 次验证；多卡共同分担每次全局 batch。ImageCHD 聚合全局有效体素和 Dice 统计，1282 按全局有效病例归约边际损失。抽样种子由全局更新序号和样本位置决定，worker 数量不改变样本身份。

## CPU 和内存

按整组读取 CPU affinity、Linux cgroup v2 CPU/RAM 限额与实时可用内存，默认 CPU/RAM 比例上限 80%，主机保留 4 GiB。协调进程、各训练进程、供数进程、重采样工作区、缓存、预取和传输输入均计入预算。

`workers: auto` 根据预算增加供数进程，默认 `max_workers: 32`，可修改；`workers: 0` 串行供数，整数表示整组上限。供数进程持续驻留，内部单线程，队列有界。缓存为各供数进程的 RAM 缓存，容量在统一预算中计算；不会自动生成全量预处理副本。

`prefetch` 和 `readers` 是可调上限，实际深度保证供数进程有任务，并受内存预算限制；`readers: auto` 默认最多 8 个病例构建名额。多个本项目任务共用 `work_root/resources` 登记预算与显卡占用。运行中自动调整 worker、缓存、预取和病例构建并发，保留 GPU 模型与全局采样游标。安全预留触发时保存完整状态、释放本组并缩小供数重新准入，默认最多三次。详细预算、等待、GPU 卡数调整及验证边界见 [资源调度](RESOURCE_SCHEDULING.md)。

异常会直接终止同组训练；恢复使用最近完整检查点，尚未提交的更新会重算。只处理本任务已登记的进程，外部任务保持运行。

## 使用旧数据划分

`split.path: null` 使用审计清单中已有角色。指定 `--split /绝对路径/splits.json` 或配置 `split.path` 后，按该文件选择病例和角色，不重新随机生成名单。

| 数据集/格式 | 字段 |
|---|---|
| ImageCHD 旧 splits.json | `train`、`val`、`test`、`aux_train` |
| CHD1282 角色格式 | `train`、`development`、`test` |
| CHD1282 原始 cohort.json | 协议 `dataset1282-scheme-a-preprocess-1`，`cases` 中的患者与角色 |
| ImageCHD 汇总 splits.json | `roles: {case_id: {"0": role, "1": role, "2": role}}` |
| nnU-Net 折列表 | `[{"train": [...], "val": [...]}]` |

使用 `--fold 0` 选择折；折列表必须包含相应索引。每折独立文件使用对应文件。可以手动修改名单或指定其他文件。若旧病例编号对应清单的 `source_key`，明确设置 `--split-id-field source_key`；默认精确匹配 `case_id`。

ImageCHD 的辅助病例参与训练，保留 `cohort: auxiliary`，预处理统计仅拟合主训练病例。验证与测试病例均不参与统计拟合。旧 ImageCHD 使用病例级独立性，`split.unit: auto` 对其保留病例级语义，不推断患者身份；已有可靠患者映射时可显式选择 `unit: patient`。1282 始终使用患者级划分。未知病例、重复角色和划分内容改变直接报错。准备和训练必须使用相同的 split、fold 设置。

1282 原角色规则为种子 `20261104`，按最早合格儿童扫描的来源设备及年龄层分组，组内使用患者摘要排序，以最大余数方法分配 70/15/15，同患者重复扫描继承角色。导入保留 train、development、source_test、target_primary、target_secondary 五个角色；后续审核名单中的排除和索引扫描标记同样生效。测试组不参与统计拟合或训练。原 Scheme A 使用固定 fold 0，手动三折可另选明确的折列表。内部、外部两个测试入口及集中迁移方式见 [测试与输入迁移](TESTING_AND_TRANSFER.md)。

ImageCHD 原流程先对完整标注的历史 97 例按种子 42 排序，分配三折测试，再按 `42 + 1009 × (fold + 1)` 分配验证，最后排除已确认问题病例。导入文件直接保留旧角色；fold 0 为主训练 53、辅助训练 13、验证 10、测试 33。不同折从原汇总角色文件或各折文件读取。

划分文件和患者清单保留在私有 course 目录，不随公开代码上传。原始清单仍需包含经过审计的来源文件、摘要、逐例间距与缺标签信息；cohort.json 只补充角色和患者分组。

## 服务器用法

在已激活的环境中，从项目根目录执行。将私有配置中的 manifest、split.path 指向服务器已有审计清单和旧划分文件，设置 `COURSE` 为配置中的过程目录后，准备每折描述：

```bash
python -B run.py prepare --config configs/experiments/imagechd_stunet.yaml \
  --output "$COURSE/prepared/imagechd_fold0.json"
```

自动选择两卡、全局 batch 4、200 epoch：

```bash
python -B run.py train --config configs/experiments/imagechd_stunet.yaml \
  --prepared "$COURSE/prepared/imagechd_fold0.json" \
  --device cuda --gpus 2 --batch 4 --epochs 200 --workers auto --no-share-gpu
```

手动选卡将 `--gpus 2` 替换为 `--gpu-ids 0,2`；修改 batch 使用 `--batch 8`。修改后的训练配置是新任务，需选择新的 run_dir。继续同一任务追加 `--resume`。命令行设置 split/fold 时，prepare 与 train 均传入相同设置。GPU 预测单次使用一张通过空闲准入的显卡；共享容量证据按训练阶段记录。

## 有界验证

`tests/verify_parallel.py` 使用用户提供的一至两个真实病例、32³ 技术区域、全局 batch 4，每个进程配置最多两次更新和一次验证，比较 1–4 进程的权重变化及恢复状态。临时文件在指定 work-root 中，结束后自动清理，仅在指定 report 保存摘要。默认 CPU/Gloo；CUDA/NCCL 需在目标 Linux 环境选择 `--device cuda` 执行。正式 200 epoch 和持续吞吐不属于短测验收范围。
