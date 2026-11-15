# 离线数据合同与完整流程

合同：`chd-offline-npy-v1`。STU-Net 固定转换：`stunet-ct-nnunet-bounded-v1`。

## 版本与任务边界

在线基线公开冻结分支为 `archive/chd-online-preprocess-20261115`，固定公开提交 `2269b26a4abc8a4952acacfab0611af4ce375d4f`；对应本地独立仓库提交 `60d6e5f17c81f1fea78690021fbf403b1ee61f37`。该分支包含完整代码、pyproject.toml、正式配置和恢复协议。保留原 Python 环境、私有清单、计划、prepared、run_dir 和 latest 检查点，即可用该分支继续原在线任务。

离线版本使用新的 prepared 和 run_dir。程序拒绝把旧在线描述作为离线训练输入；旧任务与新数据身份分别记录。现有精确源码兼容表继续保留历史记录，不增加在线到离线的通用放行。助手2/3各自维护独立源码、分支与更新入口；适配通过确定版本的本合同/CLI，不复制或覆盖其目录。

## 阶段与模块

1. `prepare`：读取冻结清单/划分，只用当前折的主训练组拟合间距和灰度统计，保存来源描述。
2. `preprocess`：校验来源，按内存额度并行执行整例固定转换，保存经过读回核验的离线数据和完整描述。
3. `train`：只读取完整离线数据，通过 MONAI Dataset 和只读 NumPy mmap 随机取块；当前数学计算及随机抽样协议保持相同。训练期间不重新读取原图数组或做整例重采样。
4. `predict/evaluate/test`：预测读取离线影像，沿冻结计划完成滑窗和原空间恢复；评价独立读取来源标签。新预测输入需要先基于冻结训练计划离线处理，缺少对应数据时明确拒绝。

`preprocessing/validation.py` 负责来源 SHA256、尺寸、空间、间距和类别核验；`models/stunet/preprocess.py` 负责 nnU-Net 裁剪/CTNormalization/重采样与前景位置。离线存储与执行分别在 `preprocessing/offline_store.py` 和 `preprocessing/offline.py`，训练加载器仅依赖离线数据合同。

当前固定转换、网络、损失、部分标注、batch 和采样方法保持原工程语义。没有新增未定义的随机增强；后续额外增强应显式配置和验证。

## 身份与目录

```yaml
preprocessing:
  root: /data4/zhangruichen/data/chd-segmentation/preprocessed
  statistics_source: main-train
```

数据根目录下的 `index.json` 将完整配方身份映射到首次生成目录。例如：

`ImageCHD-STUNet-20261115/fold_0/<配方摘要前16位>/<病例摘要前16位>/`

目录中的短摘要便于跨平台使用，catalog 与 case.json 核对完整 SHA256，短摘要冲突会拒绝。日期用于识别首次生成批次；换日期再次执行通过 index 查找同一配方，继续复用。

配方身份包含合同版本、数据集、完整有效计划、标签定义、fold、formal/technical 范围、技术区域、明确统计来源、拟合病例的输入摘要/几何/标签规则和转换依赖版本。计划生成耗时等记录不参与配方身份。病例身份还包含原图/标签摘要、原作者间距、空间规则、缺标和忽略标签处理。GPU 数量、batch、轮数、任务名称、日志修改和日期不会单独改变固定数据配方。固定转换算法改变时必须更新处理合同。

ImageCHD fold0/1/2分别拟合各自主训练组并独立保存计划和数据。只执行命令指定的一折；不自动生成其他折。验证、内部测试和外部测试不参与统计拟合。Q/A 等不同统计来源明确设置不同 statistics_source，不能混用拟合来源。CHD1282 保留原患者级划分及两种测试入口。

## 每例数据

| 文件 | 内容 |
| --- | --- |
| `image.npy` | 转换后的影像；保持转换输出 dtype，只读 mmap |
| `label.npy` | 有标签病例的转换后标签；保留 ignore 与部分监督语义 |
| `locations.npz` | 有界前景位置数组，禁止 pickle |
| `case.json` | 完整身份、文件摘要/标记、dtype/shape、空间恢复属性、原尺寸、原 affine、工作 affine、ROI、缺标及处理耗时 |
| `owner.json` | 当前病例与配方归属，用于未完成产物的安全恢复 |

case.json 中前景类别键以独立列表保存，加载时恢复整数或 tuple。JSON 中空间属性使用普通数值/列表，数组存储使用 NumPy 成熟格式。

写入先在同级 `.pending` 病例目录完成，逐数组独立读回、数值比较、fsync 和摘要核验通过后才原子更名。中断后只处理归属确认的未完成目录。重跑逐例核验已有文件，完整病例复用；缺失或摘要/标记核验失败的已归属病例重新生成。不同输入与配方不覆盖旧版本。数据仍被活动任务使用时禁止修补。

catalog 记录完成病例；离线 prepared 声明本任务需要的病例视图、数据目录与完成身份。全部所需病例完成后才提交描述，训练准入核验描述、配方和各病例完成记录。训练进程内再次核对持久文件标记，运行中改变文件立即报错。写入耗时和文件时间标记不进入病例内容身份，按同样内容修复后可以保持原身份。

## 资源和日志

预处理单独使用 CPU，worker 内部数值库单线程；并发受 workers/max_workers、readers/max_readers、CPU 比例、主机及 cgroup 余量与各病例工作空间之和约束。大病例占用额度期间，小病例满足条件仍可处理；预算不足持续等待。每个病例完成后保存 catalog，重新启动只补缺失或核验失败的病例。

训练按 mmap 取块和前景位置内存核算预算，免除原整例重采样工作空间预留。mmap 的文件总字节与实际驻留 RAM 分别记录，操作系统文件页缓存仍会使用真实内存。各 worker 的 LRU 可缓存映射及位置数组；未命中时重新打开离线文件，不重新预处理。供数当前批容量为1，全局 batch4按选定进程分担；主机就绪后才进入 DDP。

显卡默认只选择空闲卡，数量不足持续等待；自动/手动选卡、固定 batch、可选最少卡数、完整 epoch 边界重组和完整恢复继续沿用。没有对应实测容量时不承诺增加卡数或显存足够。检查点包括模型、优化器、scaler、进度、已消费游标和逐 rank RNG；保存前后读回核验。

过程日志在 run_dir/events.jsonl，有界轮转。预处理打印逐例读取、转换、保存、复用、总数/已完成数、资源额度和耗时。训练记录轮次、attempts/updates/skipped、loss、验证、供数/计算耗时、资源分配与检查点；心跳只表示进程仍在当前阶段。

## 引用和保留

训练及预测在数据版本的 references.json 登记 run_dir、模型、数据身份、进程出生身份和活动状态，结束后保留历史引用。`data-info --prepared <离线描述>` 给出实际容量、引用及可清理范围；活动引用或多个模型引用时标记清理受阻。

正式数据不自动删除。用户确认不再恢复或预测后，可依据精确版本目录管理数据；删除范围仅限指定模型/折/配方，不包含原数据、其他折和其他模型。共享数据仍有其他模型引用时必须保留。任务日志、最新完整续跑点、最终掩膜及评价分别保存在 course/result。

## zmic 命令：独立一折

先使用已激活的环境。更新命令使用交付回执中的确定提交及 `--force`；活动任务退出后才能切换代码。

```bash
CODE=/data4/zhangruichen/code/chd-segmentation
PY=/data4/zhangruichen/env/chd-segmentation/bin/python
COURSE=/data4/zhangruichen/course/chd-segmentation
CFG="$COURSE/inputs/imagechd_stunet_fold0.yaml"
NAME=ImageCHD-STUNet-20261115_fold0
RUN="$COURSE/runs/$NAME"
SOURCE="$COURSE/prepared/${NAME}_source.json"
READY="$COURSE/prepared/${NAME}_offline.json"
DATA=/data4/zhangruichen/data/chd-segmentation/preprocessed

"$PY" -u -B "$CODE/run.py" prepare --config "$CFG" --run-dir "$RUN" --output "$SOURCE"
"$PY" -u -B "$CODE/run.py" preprocess --config "$CFG" --run-dir "$RUN" \
  --prepared "$SOURCE" --output "$READY" --root "$DATA"
"$PY" -u -B "$CODE/run.py" train --config "$CFG" --run-dir "$RUN" --prepared "$READY" \
  --device cuda --gpus 2 --min-gpus 2 --batch 4 --epochs 200 --workers auto \
  --no-share-gpu --wait-seconds forever
```

继续此离线任务时在 train 命令添加 `--resume`。后台运行可用 nohup 将上述顺序命令的输出追加到 `$COURSE/logs/${NAME}.log`。fold1/2换对应私有配置与 NAME，不更改原清单名单。1282 使用其对应配置和 `CHD1282-STUNet-日期_fold0` 名称。

```bash
"$PY" -B "$CODE/run.py" data-info --prepared "$READY"
"$PY" -u -B "$CODE/run.py" predict --config "$CFG" --run-dir "$RUN" \
  --output "/data4/zhangruichen/result/chd-segmentation/$NAME/predictions"
"$PY" -u -B "$CODE/run.py" evaluate --config "$CFG" --run-dir "$RUN" --prepared "$READY" \
  --predictions "/data4/zhangruichen/result/chd-segmentation/$NAME/predictions" \
  --output "/data4/zhangruichen/result/chd-segmentation/$NAME/metrics.json"
```

预测需要完成的训练检查点。批量预测范围由清单和 --case 决定；`test --set internal/external` 保留独立测试入口。
