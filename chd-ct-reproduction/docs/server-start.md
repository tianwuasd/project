# zmic44 服务器启动

当前所有服务器入口都调用 ImageCHD 主流程。旧论文入口位于 back/paper-v1。

## 配置来源与环境

配置文件 configs/servers/zmic44.json 对应指定工作笔记 `zmic44_服务器基础信息_更新版9.22.md` 的实际更新时间 **2026-09-26**。

记录为 Ubuntu24.04.3、2×AMD EPYC7742（128核/256线程）、251GiB内存、8×RTX3090 24GiB、驱动580.173.02。以上为历史快照；脚本启动时重新检查GPU，不固定沿用某张历史空闲卡。

- 复用已有 Conda 管理工具：`/data5/zhougaowei/zhangruichen_workspace/project1_runtime/tools/miniforge3/bin/conda`。
- CHD独立环境：`/data5/zhougaowei/zhangruichen_workspace/chd_ct_runtime/envs/chd_py311`。
- 默认预处理目录：`/data_nas/zhangruichen/chd_ct_data/imagechd7`。
- 默认模型/日志/预测结果：`/data_nas/zhangruichen/chd_ct_results`。
- 默认2个计算线程。已有 project1 的 Python 仅记录为参考，不会在其中安装CHD依赖。

新环境使用 Python3.11、PyTorch2.8/CUDA12.8；优先使用已有 Conda，找不到时才下载校验过的 Miniforge。驱动显示的 CUDA13.0 是支持上限，不等同环境里的 PyTorch CUDA runtime。大数据放 NAS，避免挤占此前使用率较高的 /data5。

## 四个独立命令

在服务器的 `project/chd-ct-reproduction` 目录执行。下方原始数据路径需要替换为实际上传/解压位置。

```bash
bash preprocess_server.sh --dataset /实际路径/ImageCHD_dataset --non-interactive
bash train_server.sh --mode smoke --non-interactive
bash train_server.sh --mode train --non-interactive
bash predict_server.sh --models /完整路径/train --split test --non-interactive
bash evaluate_server.sh --predictions /完整路径/predict --split test --non-interactive
```

正式训练先做相同模型尺寸的 preflight；失败就停止。预测的模型目录必须包含 models.json 和全部六阶段权重。若使用短测模型，预测加 `--allow-smoke`。

也可运行 `bash start_server.sh`，按菜单选择环境、预处理、训练、预测、评估或合成测试。默认回车只做环境检查。无人值守运行必须明确 `--task` 或使用上述功能脚本。

```bash
bash start_server.sh --task environment --non-interactive
bash start_server.sh --task demo --device cpu --non-interactive
bash train_server.sh --mode check --non-interactive
```

预处理、缓存检查、评估使用CPU。训练/预测默认使用单张空闲GPU，尊重 CUDA_VISIBLE_DEVICES；没有合适GPU时停止，不越过已分配范围。`--gpu auto/物理编号/完整UUID` 可指定选择方式。

## 可覆盖设置

支持 `--runtime`、`--results`、`--prepared`、`--python`、`--server-config`。使用另一个缓存目录时，每个步骤传入相同的 `--prepared`。`--python` 表示直接复用已有环境并跳过安装，仍执行环境检查。

例：已有96³缓存可直接指定，无需重新预处理：

```bash
bash train_server.sh --prepared /data_nas/zhangruichen/chd_ct_data/imagechd7_96 --mode smoke --non-interactive
```

每次任务生成独立时间戳目录，包含 status.json、server.log 和各阶段输出。训练输出在该目录的 train 子目录，预检在 preflight，预测在 predict。预处理缓存输出在 --prepared 指定目录。运行中的同一 runtime 由文件锁保护；中断会清理其子进程组。

脚本前台运行，长训练可放在 tmux 中。远程网络设置沿用服务器自身环境，不把本机127.0.0.1代理写入远程配置。

本次已验证本机CPU与模拟派发，尚未连接zmic44执行安装、CUDA预检或正式训练。
