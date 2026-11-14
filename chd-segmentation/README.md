# 先心分割处理

版本 0.1.0。当前接入 STU-Net-S 的 CHD1282（11 类）和 ImageCHD（8 类）流程：准备描述、训练、完整状态恢复、MONAI 滑窗预测、原数组网格导出、独立评价和短测。两数据集共用网络代码，分别训练权重。训练支持 CPU、1–4 卡 CUDA/DDP、自动或手动选卡、固定全局 batch 分担和有界 CPU 供数。

1282 保留训练、验证、内部测试及两个外部设备子组，通过 `test --set internal`、`test --set external` 两个入口运行。旧审核排除、患者索引及冻结角色一并读取。私有清单和配置可集中迁移，再用 `bind-inputs` 绑定服务器既有数据；具体命令见 [测试与输入迁移](docs/TESTING_AND_TRANSFER.md)。

各阶段实时记录进度、耗时、资源、复用和异常；长步骤默认每 15 秒输出心跳，详细日志统一保存在过程目录的 `run_dir/events.jsonl`，并限制轮转容量。设置与查看方式见 [运行日志](docs/LOGGING.md)。

短测限定每数据集一例真实病例、32³ 技术子区域、每组合两次更新，包含保存后继续更新。CPU/Gloo 的 1–4 进程分担、数值一致性、完整恢复和预测复用已验证；目标 Linux CUDA/NCCL、共享显卡实况、正式尺寸容量和持续性能待服务器验收。prepare 默认根据当前主训练清单生成预处理计划，短测生成独立技术计划。资源参数和旧划分读取见 [GPU/CPU 使用说明](docs/GPU_CPU_USAGE.md)。Xu、MedSAM2、Q 的后续接入见整体开发计划。

## 环境与入口

使用独立 Python 3.11+ 环境安装本目录的依赖，再执行：

```text
python -B setup_env.py --check
python -B run.py --help
python -B run.py --version
```

本机的 `.venv` 复用既有研究依赖并安装 MONAI 1.5.2；其本地依赖引用不用于服务器部署。服务器需要独立安装 `pyproject.toml` 声明的依赖并验证对应 PyTorch 构建。代码不自动安装依赖，也不自动更换已有环境。

zmic 使用 `/data4/zhangruichen`：项目实体目录固定在 `code/chd-segmentation`，包含全部入口、src、configs 和 docs。Git 元数据复用 `code/git/.git`，`code/.git` 关联该元数据；本项目仅同步 chd-segmentation，其他项目文件保留。过程产物在 `course/chd-segmentation`，最终结果在 `result/chd-segmentation`，独立环境计划放在 `env/chd-segmentation`，原始数据从 `data` 读取。部署、配置及更新命令见 [zmic 部署说明](docs/ZMIC_DEPLOYMENT.md)。

独立 `update.py` 只使用 Python 3.11+ 标准库与系统 Git。它取得本项目需要的文件对象，检查语法，再将本地 main 快进到固定的远端提交；成功后 HEAD、origin/main 和核对时的远端 main 一致，并逐文件核验本项目内容。`run.py --code-info` 显示实际代码、提交与解释器，本地修改标记只检查本项目。运行任务请统一使用 `run.py` 或 `smoke.py`，它们持有代码锁；本项目的本地修改、未知文件或活动任务会阻止更新。其他项目的工作文件保留；若其暂存内容与远端提交冲突，Git 会停止更新并保留暂存内容。更新不会安装环境或启动训练。

公开仓库更新默认探测 `ghfast.top`、`gh-proxy.com` 与 GitHub，选择提交匹配、支持按需下载且响应较快的镜像。地址和超时保存在 `course/chd-segmentation/update_network.json`，可以手动修改；也可使用 `--mirror 地址` 或 `--mirror direct`。镜像请求不携带 Git 凭据，选定来源与探测结果记录到过程目录。文件内容校验失败时停止本次更新。

用户明确允许本项目以远端为准时，执行 `python3 -B update.py --force`。它修复部分索引、本项目暂存删除和缺失文件，并允许同步到重写后的远端 main；目标源码的本地修改会被覆盖。首次切换执行 `--force --relocate`，持有任务锁后将实体项目从 code/git/chd-segmentation 移到 code/chd-segmentation，再同步代码。数据、环境、运行结果、其他项目和非目标未知文件保留。

## 日常更新

服务器同步 GitHub main 的最新代码：

```bash
python3 -B /data4/zhangruichen/code/chd-segmentation/update.py
```

默认不启用强制同步；需要覆盖本项目受 Git 管理的本地代码修改时，在命令末尾添加 `--force`。该参数只对本次执行生效。更新成功后，回执中的 `sha`、`head`、`origin_main` 应一致，`tracked_code_clean` 应为 `true`。

## 准备私有输入

实验配置示例在 `configs/experiments/`，机器路径集中填写。每套数据需要官方 `small_ep4k.model` 以及私有清单。相对病例路径按清单的 `data_root` 解析。默认 `planning.mode: auto`，无需提供已有计划；prepare 仅使用清单中 train 病例拟合灰度统计及目标间距。计划保存在 work_root 下的 `plans/<dataset>/fold_<fold>/<scope>/preprocessing_plan.json`，也可显式设置 plan 输出路径。配置、身份复用及旧计划使用方式见 [自动预处理计划](docs/PREPROCESSING.md)。

`work_root` 限定预处理描述、运行日志和检查点；可选 `result_root` 限定预测与评价输出。省略 `result_root` 时沿用 `work_root`；zmic 示例已显式设置两个独立目录。私有配置保存在 `course/chd-segmentation/configs`，清单中的 `data_root` 填写 `data` 内实际数据集位置。示例中的输入文件位置需核对，不能直接代表服务器已经具备这些文件。

```yaml
dataset: imagechd
scope: formal
plan_scope: formal
data_root: /path/to/raw-data
cases:
  - case_id: example_case
    image: images/example_case.nii.gz
    label: labels/example_case.nii.gz
    patient: private_patient_key
    roles: [train]
    spacing_xyz: [0.3, 0.3, 0.75]
    unknown_classes: [5]
    known_absent_classes: []
    geometry_policy: matching_affine_author_spacing
    ignore_label_ids: []
    sha256:
      image: actual_image_sha256
      label: actual_label_sha256
```

正式训练还需独立 validation 病例。1282 按患者划分；ImageCHD 导入原划分时保留病例级独立性，有可靠患者映射时可明确要求患者级检查。示例数字仅解释字段，实际值必须来自审核记录。ImageCHD 的未知类别仅允许缺 MYO（编号 5）；1282 使用其原始 0—10 编号和 `shared_index_excel` 空间规则。未标注和已确认缺如分别记录。额外标签仅按显式 `ignore_label_ids` 屏蔽。plan_scope 仅用于显式读取既有计划的来源声明，不能代替对拟合病例的核查。

图像和标签原件只读。prepare 按病例顺序读取主训练数据，在 RAM 中进行有界前景采样，保存紧凑计划与描述，不输出整套重采样副本；训练和预测使用 MONAI Dataset/Compose 在 RAM 处理。持久供数进程使用各自有界 LRU 缓存，按整组预算核算；CPU 供数、缓存、读盘并发和预取在安全边界按实时余量调整。默认只使用空闲显卡，卡数不足持续等待；显式设置最少卡数可启用 epoch 边界的 GPU 卡数调整。详见 [资源调度](docs/RESOURCE_SCHEDULING.md)。

## 运行

```text
python -B run.py prepare --config /data4/zhangruichen/course/chd-segmentation/configs/imagechd_stunet.yaml --output /data4/zhangruichen/course/chd-segmentation/prepared/imagechd_fold0.json
python -B run.py train --config /data4/zhangruichen/course/chd-segmentation/configs/imagechd_stunet.yaml --prepared /data4/zhangruichen/course/chd-segmentation/prepared/imagechd_fold0.json
python -B run.py train --config /data4/zhangruichen/course/chd-segmentation/configs/imagechd_stunet.yaml --prepared /data4/zhangruichen/course/chd-segmentation/prepared/imagechd_fold0.json --resume
python -B run.py status --run /data4/zhangruichen/course/chd-segmentation/runs/imagechd_stunet_fold0
python -B run.py predict --config /data4/zhangruichen/course/chd-segmentation/configs/imagechd_stunet.yaml --output /data4/zhangruichen/result/chd-segmentation/imagechd/fold0/predictions
python -B run.py evaluate --config /data4/zhangruichen/course/chd-segmentation/configs/imagechd_stunet.yaml --prepared /data4/zhangruichen/course/chd-segmentation/prepared/imagechd_fold0.json --predictions /data4/zhangruichen/result/chd-segmentation/imagechd/fold0/predictions --output /data4/zhangruichen/result/chd-segmentation/imagechd/fold0/metrics.json
```

预测通过 `--inputs` 读取相同数据协议的新清单，通过 `--case` 选择病例。推断不读取标签数组；评价独立执行。导出默认恢复原始数组尺寸与 source affine，同时在结果记录保留工作毫米网格的来源。选择 `export_space: working` 时使用衍生毫米网格。新任务应使用新目录，完成结果按身份复用。

训练配置支持固定尝试预算或 epoch 预算、SGD、深监督及显式学习率；正式示例默认 200 epoch，每轮 250 次训练尝试、50 次验证，使用 poly 学习率。`save_every_attempts` 控制周期保存，轮次边界、正常停止与结束都会保存。最新完整状态包含模型、优化器、scaler、逐进程随机状态、消费和验证进度；恢复拒绝缺失检查点、代码或输入身份变化。当前保留 latest，不提供 best 选模或旧工程检查点导入。

## 短测与保留

```text
python -B smoke.py --spec /data4/zhangruichen/course/chd-segmentation/configs/smoke.local.yaml --work-root /data4/zhangruichen/course/chd-segmentation/smoke/unique_task --report /data4/zhangruichen/course/chd-segmentation/smoke-reports/unique_task
python -B -m unittest discover -s tests -v
```

`project.local.yaml` 是私有技术短测说明，包含最多两个真实病例。短测明确采用技术子区域和角色重叠，仅验证接口；不得用其指标报告正式模型效果。它调用相同的 prepare/train/predict/evaluate，实现真实保存、重载、继续更新、预测复用、源文件摘要复查和自动清理。

`--validate-only` 只验证初始化状态保存/恢复与预测，不执行优化器更新。单元测试中的操作系统锁检查需要将 `CHD_TEST_ROOT` 指向获准的测试目录，临时锁目录自动清理。

正式训练每阶段保留一份最新完整检查点、运行配置、进度和有界事件日志。预测保留每例最终掩膜及摘要，概率不持久保存。短测只保留输入身份、实际结果、阶段耗时、资源峰值和清理记录；临时配置、权重和掩膜在本任务退出时清理。

预测事件日志追加到对应训练运行的 `events.jsonl`；最终结果目录只保存掩膜、必要身份/完成记录和写出锁。短测在自有临时目录验证过程/结果分离，全部测试掩膜随该目录清理。

## 来源

STU-Net-S 结构源自 uni-medical/STU-Net，Apache-2.0，上游文件 blob `85587aca5fdf1180dc266a138f3ae97c231875b2`。骨干按固定官方权重 SHA256 加载，输出头独立初始化。部分监督分别保留原 ImageCHD 与 1282 工程的计算方式，详细依据见随交付保存的开发计划。
