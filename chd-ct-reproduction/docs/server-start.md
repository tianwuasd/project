# zmic44：按步骤运行 CHD

服务器入口已支持数据默认路径、记住设置、选择 1—5 张 GPU、独立预处理/短测/训练/测试。复用 whole_heart/project1 的个人环境、多卡任务队列和日志组织思路，CHD 继续使用自己的六阶段模型、七结构标签及配置。

## 先改哪一个文件

集中设置在 `configs/servers/zmic44.json`。第一次也可以直接运行 `bash start_server.sh`，按中文提示设置；回车选择默认值。

| 设置 | 默认值 |
|---|---|
| `dataset` 原始数据 | `/data_nas/zhangruichen/baidu_import/ImageCHD_dataset` |
| `prepared` 缓存 | `/data_nas/zhangruichen/chd_ct_data/imagechd7` |
| `results` 模型、日志、预测、评估 | `/data_nas/zhangruichen/chd_ct_results` |
| `runtime` 环境与下载缓存 | `/data5/zhougaowei/zhangruichen_workspace/chd_ct_runtime` |
| `config` 模型设置 | `configs/chd.yaml` |
| `gpu_count` 训练 GPU 数 | `1`，可设 1—5 |
| `gpu` 指定卡 | `auto`；也可写 `0,2`，数量须与 gpu_count 一致 |

原始数据默认路径是本项目约定位置，尚未确认已上传；数据必须已解压，符合 ImageCHD 的 image/label 配对规则。不能把 Wholeheart_Train_Dataset 直接当 ImageCHD 输入。路径不正确时用 `--dataset /实际路径/ImageCHD_dataset` 覆盖。

设置优先级：**命令行 > runtime 中记住的设置 > 服务器配置文件**。保存在 `<runtime>/.server-settings.json`，每次运行另存 `server-settings.json` 快照。更换配置默认值后，用 `--reset-settings` 忽略上次选择；不删除缓存或模型。模型结构和 gate 开关仍在 CHD YAML 中。

## 每一步单独运行

在服务器的 `project/chd-ct-reproduction` 中执行。所有命令前台运行，长训练可放在 tmux 中。

```bash
# 0. 环境检查；逐张检查计划使用的显卡
bash start_server.sh --task environment --gpu-count 2 --gpu auto --non-interactive

# 1. 预处理：使用上述默认原始目录；仅生成缓存与患者划分
bash preprocess_server.sh --non-interactive
# 数据在其他位置时：
# bash preprocess_server.sh --dataset /实际路径/ImageCHD_dataset --non-interactive

# 2. 短测：缩小网络，每阶段一个训练 batch 和一个验证 batch
bash smoke_server.sh --gpu-count 2 --gpu auto --non-interactive

# 3. 正式训练：先做真实配置尺寸的 preflight，通过后才开始正式训练
bash train_server.sh --mode train --gpu-count 2 --gpu auto --non-interactive

# 4. 测试：复用上次成功训练的模型，默认仅预测并评估 test 划分
bash test_server.sh --non-interactive
```

需要指定卡时，使用如 `--gpu-count 2 --gpu 0,2`；仍会检查这些卡是否处于允许范围且空闲。缓存目录改动时传 `--prepared /新缓存目录`，后续命令会记住。预处理输出已存在会停止，不覆盖或重复处理。

`smoke_server.sh` 只验证训练能否跑通；`test_server.sh` 对已有模型做测试集推理和 Dice 评估，两者含义不同。只有短测权重时，明确使用 `bash test_server.sh --allow-smoke --non-interactive`；短测结果不能代表正式模型精度。

选择具体模型或拆开预测和评估：

```bash
bash test_server.sh --models /完整路径/某次运行/train --split test --non-interactive
bash predict_server.sh --models /完整路径/某次运行/train --split test --non-interactive
bash evaluate_server.sh --predictions /完整路径/某次运行/predict --split test --non-interactive
bash train_server.sh --mode check --non-interactive
```

训练只读缓存，不预处理、不运行测试集评价。测试不训练、不预处理。无标签 CT 使用独立预测，不能运行需要真值的测试/评估。正式训练失败不会把不完整模型设置为上次成功模型。

## 多卡怎样工作

沿用 whole heart 的**一张卡一个模型任务**方式，不使用 DDP，也不把多张显卡显存合并。

- `crop64`、`crop128`、`all64`、`all128`、`blood2d` 可独立训练，最多五个同时运行。
- `blood_lstm` 必须等待本次 `blood2d` 成功，使用并冻结其编码器。
- 子进程只可见自己分配的 GPU；开始下一阶段前重新检查该卡。任何阶段失败会停止新派发并结束本队列仍在运行的子进程，保留日志与已经完成的权重。
- 同时运行的数量受 GPU 数和可运行阶段数共同限制，后期可能只有一张卡在工作。
- 预处理、缓存检查和评估用 CPU。融合预测/测试用一张卡；若保存了多个显式编号，使用列表第一张。`gpu_count` 控制训练并发数及环境检查卡数。
- 每个训练阶段采用 `seed + 阶段序号`，使单卡顺序运行与多进程运行有一致的随机种子。它改变旧版本后续阶段的随机数轨迹，但不改变网络或已有权重的加载格式。

保持尊重 `CUDA_VISIBLE_DEVICES`；空闲卡不足、卡上已有计算进程、低于 6000 MiB 空闲显存或利用率高于 10% 时停止，不自动降卡数或切到 CPU。空闲检查不是集群调度器预约，共享服务器仍应先确认分配。

## 输出在哪里

每次步骤生成 `<results>/<时间戳>_<标识>/`：

```text
status.json                 # 本次任务状态
server-settings.json        # 本次设置快照
server.log                  # 环境检查与派发记录
environment/0/              # 第 1 张卡的实际计算检查（其他卡依次编号）
preflight/                  # 正式训练前的完整尺寸短测
train/                      # 训练权重、models.json、各阶段 history
  stage_logs/               # 多卡时每阶段的进程日志
  queue.json                # 多卡依赖/状态/GPU 记录
test/                       # 只有 test 任务生成
  predictions/              # 分割 NIfTI 与预测记录
  evaluation/evaluation.json
  test-report.json
```

每个任务只生成自己对应的目录。只有六阶段全部完成，模型集合才标记 complete。预处理缓存仍输出到 prepared 指定的位置，不放进每次时间戳目录。中途失败的任务目前不支持从断点继续；保留现场后重新启动会创建新的任务目录。

## 环境来源与验证范围

参考工作笔记 `zmic44_服务器基础信息_更新版9.22.md`，实际更新时间为 **2026-09-26**：Ubuntu 24.04.3、8×RTX3090 24 GiB、驱动 580.173.02、251 GiB 内存。硬件信息是历史记录，脚本会重新查询显卡。

复用已有 Conda 管理工具 `/data5/zhougaowei/zhangruichen_workspace/project1_runtime/tools/miniforge3/bin/conda`；CHD 使用自己 runtime 下的 `envs/chd_py311`，不向 whole heart 的 Python 环境安装依赖。默认 Python3.11、PyTorch2.8/CUDA12.8；未找到已有 Conda 时下载并校验 Miniforge。可用 `--python /已有环境/bin/python` 跳过安装，仍检查环境。网络设置沿用服务器环境。

本次验证包括真实 CPU 多进程训练、单进程/多进程权重一致性、独立测试入口与模拟显卡派发；未连接 zmic44，尚未做真实多 GPU 或 3090 显存验证。原作者宽度配置 `configs/chd-author-unet.yaml` 仍须在服务器做 preflight；主干来源与三维默认 gate 见 [U-Net 说明](unet-source.md)。
