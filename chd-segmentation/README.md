# 先心分割处理

版本 0.1.0。当前接入 STU-Net-S 的 CHD1282（11 类）和 ImageCHD（8 类）CPU 流程：准备描述、训练、完整状态恢复、MONAI 滑窗预测、原数组网格导出、独立评价和短测。两数据集共用网络代码，分别训练权重。

本轮实测使用每数据集一例真实缺标签病例、32³ 技术子区域、每组合两次更新，包含保存后继续更新。完整病例训练效果、正式尺寸容量、CUDA/DDP、资源自动调度和服务器更新尚未验收；GPU 配置会明确拒绝运行。已有拟合计划是当前输入要求，新计划拟合尚未接入。Xu、SAM-Med3D、Q 的接入安排保存在总开发计划中。

## 环境与入口

使用独立 Python 3.11+ 环境安装本目录的依赖，再执行：

```text
python -B setup_env.py --check
python -B run.py --help
python -B run.py --version
```

本机的 `.venv` 复用既有研究依赖并安装 MONAI 1.5.2；其本地依赖引用不用于服务器部署。服务器需要独立安装 `pyproject.toml` 声明的依赖并验证对应 PyTorch 构建。代码不自动安装依赖，也不自动更换已有环境。

zmic 使用 `/data4/zhangruichen`：`code/git` 保存 Git 对象缓存，实际运行代码在独立的 `code/chd-segmentation`，由 `current.json` 选择已校验版本；过程产物在 `course/chd-segmentation`，最终结果在 `result/chd-segmentation`，独立环境计划放在 `env/chd-segmentation`，原始数据从 `data` 读取。部署、配置及更新命令见 [zmic 部署说明](docs/ZMIC_DEPLOYMENT.md)。目录存在与环境可用性须按服务器实际文件核对。

独立 `update.py` 只使用 Python 3.11+ 标准库与系统 Git。默认只安装本项目源码并记录完整 SHA；指定 `--activate --python 实际解释器` 后执行依赖检查，通过后原子设置生效指针。受管目录的 `run.py --code-info` 显示实际代码、提交与环境；运行期间持有共享代码锁。新版本源码不会覆盖正在运行任务的文件。

## 准备私有输入

实验配置示例在 `configs/experiments/`，机器路径集中填写。每套数据需要既有拟合计划、官方 `small_ep4k.model` 以及私有清单。相对病例路径按清单的 `data_root` 解析。

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

正式训练还需独立 validation 病例，患者不能跨角色。示例数字仅解释字段，实际值必须来自审核记录。ImageCHD 的未知类别仅允许缺 MYO（编号 5）；1282 使用其原始 0—10 编号和 `shared_index_excel` 空间规则。未标注和已确认缺如分别记录。额外标签仅按显式 `ignore_label_ids` 屏蔽。

图像和标签原件只读。prepare 保存紧凑描述与计划引用，不输出整套重采样副本；训练和预测使用 MONAI Dataset/Compose 在 RAM 处理。当前缓存为本进程有界 LRU，不能宣称跨进程共享或自动 worker 扩展。

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

训练配置使用固定尝试预算、SGD、深监督及显式学习率；`save_every_attempts` 控制周期保存，正常停止与结束都会保存。最新完整状态包含模型、优化器、随机状态和消费进度；恢复拒绝缺失检查点、代码或输入身份变化。当前保留 latest，不提供 best 选模或旧工程检查点导入。

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
