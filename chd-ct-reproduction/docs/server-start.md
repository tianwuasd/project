# zmic44 服务器启动

这份入口沿用此前全心项目记录中的 Ubuntu 24.04、8 张 RTX 3090（24 GB）、580.173.02 驱动环境。硬件信息来自旧记录，本次未远程连接服务器核验。一次任务只选一张 GPU，不是八卡分布式训练。

## 第一次使用

在服务器已有的 GitHub project 仓库中更新代码，进入子目录：

```bash
git pull --ff-only
cd chd-ct-reproduction
bash start_server.sh
```

依次按提示选择：

1. 环境与缓存目录，默认 `/data5/zhougaowei/zhangruichen_workspace/chd_ct_runtime`。
2. 结果目录，默认 `/data_nas/zhangruichen/chd_ct_results`。
3. 数据清单路径；首次建议回车选 `demo`，用合成数据检查整个软件流程。

入口记住上次路径，下次仍可修改。无交互运行时建议始终明确传入 `--demo` 或 `--dataset`。

首次自动在专用目录安装 Miniforge 26.7.2-0、Python 3.11、PyTorch 2.8.0/CUDA 12.8 和项目依赖。Miniforge 安装前核对官方 SHA256。不会安装系统驱动、调用 sudo 或修改 shell 配置。网络使用服务器已有的代理环境变量；Windows 上的 127.0.0.1:7897 不能直接当成服务器代理地址。

需要安装包网络访问：GitHub、conda-forge、PyPI、download.pytorch.org。网络失败后日志会保留；不自动关闭 TLS 校验。环境安装不完整时保留目录供排查，可指定新的 runtime 重试。项目的其他依赖采用 pyproject.toml 下界约束，实际版本保存在环境报告中，并非完全锁定环境。

已有兼容环境可以跳过安装：

```bash
bash start_server.sh --python /你的环境/bin/python --demo
```

## 四种运行模式

| 模式 | 行为 |
|---|---|
| `check` | 依赖、磁盘、实际 CPU/CUDA 运算、数据完整性检查；不训练 |
| `smoke`（默认） | 检查后，合成数据八阶段训练/推理；有数据时再测试最多 4 例的缩小副本 |
| `preflight` | 上述检查全部通过后，实际配置尺寸的八阶段各跑 1 个 epoch、1 个训练 batch 和 1 个验证 batch |
| `train` | 先通过 smoke 与 preflight，再从头运行八阶段正式训练 |

```bash
# 先确认服务器能跑通；自动选择一张空闲 GPU
bash start_server.sh --demo --non-interactive

# 已准备好符合项目标签规范的数据
bash start_server.sh --dataset /你的数据/manifest.csv --mode check
bash start_server.sh --dataset /你的数据/manifest.csv --mode smoke --gpu 3
bash start_server.sh --dataset /你的数据/manifest.csv --mode preflight --gpu 3
bash start_server.sh --dataset /你的数据/manifest.csv --mode train --gpu 3
```

`--gpu` 使用 nvidia-smi 的物理编号或完整 GPU UUID。继承的 CUDA_VISIBLE_DEVICES 会限制候选范围；MIG/UUID缩写等无法识别的分配方式会拒绝选择，需使用完整 UUID。自动选择要求没有计算进程、利用率不超过 10%、至少 6000 MiB 空闲；正式尺寸预检/训练门槛为 18000 MiB。该门槛只是初筛，不能保证模型能装入显存；预检失败会停止，不会自动缩小模型或退回 CPU。这不是 GPU 调度器，仍需遵守服务器的资源分配约定。

默认 `configs/server3090.yaml` 保留现有 paper 配置的网络与输入尺寸，将所有 batch 改为 1。它是单卡候选配置，**不是原论文 batch=4 的等价训练**，目前没有 3090 实测显存数据。可用 `--config` 显式指定其他正式配置。每次预检/正式训练保存实际配置；预检权重不会进入正式训练目录。

## 数据要求与 archive

数据必须有明确的 train/val 患者划分、项目内部标签编号及全心 label。完整八阶段还需要 initial_label；缺少时短测返回“部分通过”（退出码 2），入口不会继续正式训练。

本机 archive 是原始 ImageCHD，不能直接当作本项目的完整训练集。见 [archive 检查报告](imagechd-archive.md) 和 [数据规范](data-guide.md)。入口会识别原始 ImageCHD 文件夹/分卷目录并给出解释。即使手工写出 CSV，也不能跳过标签语义与空间信息核对：数字范围检查不能验证语义。

## 查看进度与处理失败

每次在结果根目录生成新的时间戳子目录：

- `server.log`：安装和任务日志。
- `status.json`：运行中/通过/失败/部分通过/中断状态。
- `quickstart/report.txt` 与 `report.json`：环境与短测报告。
- `preflight/`：实际尺寸的短预检。
- `training/`：正式训练的模型和各阶段历史。

入口在前台运行。长任务建议放在服务器已有的 tmux 会话内；SSH 断线本身不保证任务存活：

```bash
tmux new -s chd-ct
bash start_server.sh --dataset /你的数据/manifest.csv --mode train --gpu 3
# Ctrl+B 然后 D 脱离；回来时：
tmux attach -t chd-ct
```

同一个 runtime 同时只允许一个任务。不同实验可指定不同 runtime/results，并按服务器规则分配 GPU。

训练目前没有优化器状态续训功能；中断后的结果保留，再次启动是新任务。普通失败/中断会写状态；kill -9 或机器断电可能留下 running 状态，应结合实际进程和日志判断。磁盘空间门槛为环境 20 GiB、短测结果 2 GiB、正式结果 30 GiB，都是启动前下限，不是整个任务空间保证。

## 本次验证范围

已在 Windows CPU 环境验证现有合成端到端流程、服务器入口逻辑、GPU 分配边界与预检参数；Linux shell 语法检查通过。新增 Linux 自动安装、文件锁和 CUDA 运行仍需在 zmic44 实机执行确认。不能据此声称真实数据或临床指标已复现。

官方依据：[PyTorch 2.8 历史安装说明](https://pytorch.org/get-started/previous-versions/)、[Miniforge 26.7.2-0](https://github.com/conda-forge/miniforge/releases/tag/26.7.2-0)。
