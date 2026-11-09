# ImageCHD MedSAM2 使用说明

MedSAM2 使用每个目标结构的一张轴位切片和一个框，向头脚两个方向传播。训练样本从本折训练标签生成提示。留出病例使用 `simulated_gt_single_box` 时，每类取真值面积最大的切片和包围框。该评估含提示切片，须按真值模拟提示实验单独汇报。新病例使用 `manual_box`。

## 环境

使用 `.venv_medsam2`。将官方 Git 工作目录放在 runtime 下的 `deps/MedSAM2`，固定提交为 `332f30d420f1d1b08e2a79b3ae6a602458808383`。将作者权重放在 runtime 下的 `weights/medsam2/MedSAM2_latest.pt`。离线服务器可以使用已下载的这两项。提供显式位置时，追加 `--medsam2-source "$source"` 和 `--weights "$weights"`。

官方源码采用 Apache-2.0，作者权重的使用条件见模型卡。本工程的安装配置禁用 CUDA 连通区域扩展，传播配置关闭依赖该扩展的孔洞填补。[官方源码](https://github.com/bowang-lab/MedSAM2/tree/332f30d420f1d1b08e2a79b3ae6a602458808383) · [作者权重](https://huggingface.co/wanglab/MedSAM2/blob/c0b302de5e19211e9f1425c6215f6172eb5eca7f/MedSAM2_latest.pt)

```bash
python3 setup_env.py --plan --profile training --model medsam2
python3 setup_env.py --install --profile training --model medsam2
python3 setup_env.py --check --profile training --model medsam2
```

本机 CPU 专用环境安装和检查时均加 `--cpu`。环境输出和安装进度保存到 `outputs/environment`。

## 数据准备

```bash
python3 run.py prepare --list-methods
python3 run.py prepare --method medsam2 --fold all
python3 run.py prepare --method medsam2 --fold all --resume --output "$prepared"
```

服务器默认读取历史 STU-Net-S partial 运行中的审计清单、角色和主训练病例 HU 分位数。自定义来源使用 `--source-prepared "$stu_prepared"`。按病例保存原始轴位方向的单通道 CT 和标签数组，三个折共享，灰度窗按训练折分别应用。保留 96 例主队列、13 例辅助训练和既定排除规则。部分标注病例仅使用已知前景，未标记背景及 ignore 不计入损失。病例进程使用现有 CPU/RAM 准入和独立读并发限额。

来源短时不可访问时，病例按间隔重新检查，其他病例可继续处理。多次仍不可访问时，任务进入 `waiting_io` 并定期复查。病例进程意外退出后重新核验完成标记并处理未完成项。标签内容或文件摘要不符合合同的病例进入待处理清单，必需病例缺失时训练入口会拒绝该缓存。

## 短测与训练

```bash
python3 smoke.py --stage train --model medsam2 --prepared "$prepared" \
  --fold 0 --device cpu --validation-frames 2 --output "$smoke_run"
python3 run.py train --model medsam2 --capacity-only --prepared "$prepared" \
  --fold 0 --gpus auto --max-gpus 1 --output "$capacity_run"
python3 run.py train --model medsam2 --prepared "$prepared" \
  --fold 0 --gpus auto --max-gpus 1 --background --output "$train_run"
```

短测使用 128×128、连续 2 帧和 1 个 epoch。正式默认 200 个 epoch、512×512、连续 8 帧、每段最多 3 类和全局批量 2。每个训练布局使用真实 CT 片段执行单独容量探测，检查完整反向、AdamW 更新和 GPU 显存保留量。探测结果在 `capacity.json`。正式训练每 5 个 epoch 和最终 epoch 执行整卷验证，以验证病例七类平均 Dice 选择最佳检查点。

训练片段按切片索引递增，起点在首帧有可用目标的片段中均匀选择，沿用官方训练采样器默认的顺序。框提示从首帧生成。整卷预测从提示切片双向传播。

Linux 双卡使用 `--max-gpus 2`，单卡每次累积两个序列，双卡每张卡处理一个。选卡使用可见设备编号或物理 UUID，`--gpus auto` 从可用范围选择。当前双进程同步在 Windows CPU 用 Gloo 验证，Linux NCCL、GPU 显存容量和实机换卡验收仍待完成。

双进程在启动后有 120 秒完成通信初始化。初始化期间发出停止命令，监督器会结束本任务仍未就绪的计算组。若初始化超时，监督器保留原检查点并重新启动一次。连续两次超时会进入 `needs_attention`，日志列出未就绪的进程序号。

```bash
tail -n 80 -F "$train_run/logs/run.log"
cat "$train_run/status.json"
python3 run.py train --model medsam2 --stop --output "$train_run"
python3 run.py train --model medsam2 --resume --output "$train_run" --background
```

启动参数保存在 `resolved-config.json` 和 `resolved-config.yaml`。环境、数据、提示规则及权重身份分别见 `environment.json`、`data-manifest.json`、`prompt-protocol.json` 和 `weights.json`。更新、损失分量、耗时和资源事件在 `logs/events.jsonl`。检查点先提交到 runtime 的 `recovery/medsam2`，再同步到结果目录。使用 `--recovery-root` 可指定独立存储。续跑使用同一任务冻结的代码、数据、进度及优化器状态。

`--prefetch-depth` 可选择每个进程最多提前准备的批次数，默认 1。设为 0 可测同步读取，设为 2 可比较更深的队列。监督器按当前 CPU 预算和可用内存限制新增预取。更新日志记录等待时间、队列深度、样本序号和末批补齐标记。预取不提交训练进度。续跑时可调整预取深度，样本、切片和提示框仍从检查点中的序号恢复。

`--cuda-transfer-prefetch` 供 CUDA 短测和容量检查使用，默认关闭，启用时要求非零 CPU 预取深度。后台线程将下一批的首个片段放到锁页内存。该批已就绪时，训练进程通过独立 CUDA 流提前传输，最多提前传输一个片段。样本消费时等待传输完成，验证阶段前释放预传输张量。日志记录实际预传输和消费情况，`--capacity-only` 会检查额外片段的显存需求。支持此选项的冻结短测可在续跑时用 `--no-cuda-transfer-prefetch` 恢复同步传输。正式训练使用同步传输。[PyTorch CUDA 流与锁页内存](https://docs.pytorch.org/docs/stable/notes/cuda.html)

`--shared-image-position-encoding` 让同尺寸的帧共用图像编码器的位置网格。此选项默认关闭。启用前可用 `--capacity-only` 在同一设备和任务配置下验证。支持该选项的冻结任务可以在续跑时显式启用或用 `--no-shared-image-position-encoding` 关闭，实际执行方式记录在任务配置和容量报告中。

`--activation-cpu-offload` 用于显存不足、主机内存充足的 CUDA 任务，默认关闭。编码器按块在反向时重算中间结果，其余反向所需的张量暂存在普通主机内存并按需传回 GPU。分辨率、帧数、目标数、全局批量和优化器配置均按原任务执行。启用时使用同步数据传输，按每个正式 CUDA 进程额外核对 9 GiB 主机内存，短测核对 3 GiB，均在主机保留量之外。内存不足时等待，或者在同一允许列表中选择满足条件的更小计算组。用 `--capacity-only --activation-cpu-offload` 在相同设备和配置上检查完整训练步，再以同一开关启动正式训练。[PyTorch 主机暂存接口](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.save_on_cpu) · [PyTorch 激活重算接口](https://docs.pytorch.org/docs/main/utils/checkpoint.html)

新任务先把状态与事件写到本地恢复目录，`launch.json` 的 `log` 字段给出该目录的监督进程日志路径。各训练进程的原始输出分别保存在 `control/rank-*.log`。结果目录暂时无法写入日志时，训练进度仍可从本地 `logs/run.log` 和 `status.json` 读取。同目录的 `log-sync.json` 记录同步故障，恢复后按原始字节补齐日志。

若任务结束时结果目录尚不可写，恢复写入后可以对已完成任务运行同一个 `--resume` 命令补齐日志，训练进程不会重启。

每个验证病例独立提交评分与提示摘要，并绑定验证时的检查点。中断后只复用已核验的病例结果。全部必需验证病例完成后才比较最佳检查点。

训练或验证所需的已审核文件持续不可读时，监督进程释放本任务计算进程，进入 `waiting_io`。独立探测进程每隔约 60 秒检查输入能否重新读取。输入恢复后重新核验完整摘要，再从最近检查点继续。权限错误、空文件或摘要变化保留错误记录，等待处理。

训练、验证及留出集评分读取缓存前核对文件摘要。训练取片前后、消费预取批次前和提交评分前会再检查文件身份、大小和修改时间。发现变化时保留最近检查点，报出受影响文件。完整恢复原缓存后，使用同一任务的续跑入口重新校验。

启动或续跑时，监督器先用独立进程核验本组需要的缓存。训练进程共用该轮校验结果。默认 120 秒未读取到新字节或进入下一个文件时，监督器结束校验进程，记录文件路径和已读取字节，进入存储等待。

训练取片、后台预取和训练期间的验证也使用同一无进展时限，可用 `--input-read-timeout-seconds` 调整。读缓存的进程记录当前文件和读取的切片。验证读取 CT 时，每次保留一个相邻切片块，最多 8 帧，通常不超过 8 MiB。若单帧超过 8 MiB，则逐帧读取。超时日志记录该块的原轴位起止切片。前向、反向及验证传播的模型计算本身不计入读取超时。发生超时时，监督器结束本任务当前计算组，记录文件、阶段和计算进程，进入 `waiting_io`。恢复时先试读，再完整核验摘要，从最近提交的检查点恢复。

本地与结果目录均保留最近两份检查点和最佳检查点。同步先核验文件，再原子提交 `latest.json` 中的完整索引。若本地和结果目录在同一训练位置保存了不同摘要，停止同步并报告冲突。

续跑前比较本地和结果目录已提交的索引。程序选择最新的训练位置，并核验最近两份及最佳检查点。缺失或损坏的本地文件可从同一位置、同一摘要的结果目录副本恢复。两个副本都无法核验时，任务进入 `needs_attention`，保留已有检查点和进度。加载时还会校验训练轮次、样本序号和更新次数。

## 带提示的整卷预测

```bash
python3 run.py predict --model medsam2 --training-run "$train_run" \
  --prompt-source simulated_gt_single_box --output "$pred_run"
python3 run.py predict --model medsam2 --training-run "$train_run" \
  --prompt-source manual_box --image "$ct" --prompts "$boxes" --output "$new_pred"
```

人工提示 JSON 包含 `prompt_source: manual_box`、输入 CT 的 `image_sha256` 和七项 `boxes`。每项使用 `label`（1 到 7）、零起点 RAS 轴位层号 `frame`，及原切片像素坐标 `xyxy`（左、上、右、下）。类别依次为 LV、RV、LA、RA、MYO、AO、PA。每类一个框，坐标需落在切片内。正式留出结果包含原网格 NIfTI、原始类掩膜、提示 JSON 和分结构指标。
