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

Linux 双卡使用 `--max-gpus 2`，单卡每次累积两个序列，双卡每张卡处理一个。选卡使用可见设备编号或物理 UUID，`--gpus auto` 从可用范围选择。当前双进程同步在 Windows CPU 用 Gloo 验证，Linux NCCL、GPU 显存容量和实机换卡验收仍待完成。

```bash
tail -n 80 -F "$train_run/launcher.log"
cat "$train_run/status.json"
python3 run.py train --model medsam2 --stop --output "$train_run"
python3 run.py train --model medsam2 --resume --output "$train_run" --background
```

启动参数保存在 `resolved-config.json` 和 `resolved-config.yaml`。环境、数据、提示规则及权重身份分别见 `environment.json`、`data-manifest.json`、`prompt-protocol.json` 和 `weights.json`。更新、损失分量、耗时和资源事件在 `logs/events.jsonl`。检查点先提交到 runtime 的 `recovery/medsam2`，再同步到结果目录。使用 `--recovery-root` 可指定独立存储。续跑使用同一任务冻结的代码、数据、进度及优化器状态。

每个验证病例独立提交评分与提示摘要，并绑定验证时的检查点。中断后只复用已核验的病例结果。全部必需验证病例完成后才比较最佳检查点。

## 带提示的整卷预测

```bash
python3 run.py predict --model medsam2 --training-run "$train_run" \
  --prompt-source simulated_gt_single_box --output "$pred_run"
python3 run.py predict --model medsam2 --training-run "$train_run" \
  --prompt-source manual_box --image "$ct" --prompts "$boxes" --output "$new_pred"
```

人工提示 JSON 包含 `prompt_source: manual_box`、输入 CT 的 `image_sha256` 和七项 `boxes`。每项使用 `label`（1 到 7）、零起点 RAS 轴位层号 `frame`，及原切片像素坐标 `xyxy`（左、上、右、下）。类别依次为 LV、RV、LA、RA、MYO、AO、PA。每类一个框，坐标需落在切片内。正式留出结果包含原网格 NIfTI、原始类掩膜、提示 JSON 和分结构指标。
