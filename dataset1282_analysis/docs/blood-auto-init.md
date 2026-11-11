# Blood 加速与自动近端辅助分割

`scripts/xu_pipeline.py run --resume --auto-init` 复用原预处理、固定患者划分、定位框、四路分割及 blood 完整轮次检查点，新增自动近端监督、init64/init128 与六路融合。原运行合同、原辅助标签和原跳过收据保留。

## Blood 执行

每个病例仅驻留一个经过摘要校验的 CT 压缩数组句柄和一个 blood 标签句柄。读取使用同一把锁保护压缩库访问，缩放与预取继续使用已有 CPU 队列；每次切片读取检查文件状态，病例结束释放句柄。实际读出的区域仍检查有限值、形状、类型与标签范围，变化或损坏直接报告。

时序训练保持一个病例、七张有序切片和原增强随机数顺序。冻结编码器按本卡预算分组计算这些切片，最多七张一起计算，序列网络仍执行一次原有更新。开发集没有增强，相同病例的相同切片特征可复用，最多保留七张，病例结束清空。训练增强后的跨窗口特征不复用。

根据本进程获准的显存预算决定是否启用激活重计算。显存充足时减少重计算，余量较小时保留。正式训练前的原容量检查执行相同策略与实际优化器更新，报告包含 `blood_execution`。病例日志的 `execution` 记录预算、实际七帧分组大小和重计算设置。模型参数名称、FP32、loss、全局 batch、学习率和 encoder 12 / sequence 3 轮保持原设置。

诊断对照可设置 `XU_BLOOD_EXECUTION=legacy`；默认 `auto`。`XU_BLOOD_FRAME_BATCH=1..7` 限制分组上限。`XU_BLOOD_CHECKPOINT=auto|on|off` 控制重计算；显存不足的显式 `off` 请求会交由资源等待处理。GPU 授权、子进程隔离、排除物理 0/1 和外部占用检测继续沿用原调度器。

## 自动 init 标签

默认 `--init-radius-mm 30 --init-margin-percent 20`，实际半径 36 毫米。对已标注的 AO、PA 分别寻找距离 LV/RV 联合表面最近的血管体素，以该点为中心计算物理球形范围，保留范围内与种子连通的血管部分。保留五种心脏结构并将选中的 AO/PA 近端合并为第六个前景类；未知体素继续使用 `-1`。

这是用户允许的自动近端范围估计。它不提供人工审核的瓣膜端面；先天异常情况下 AO/PA 与心室的对应由实际标签距离确定。半径和扩大比例为固定工程参数，不根据开发集效果调优。

生成只覆盖既定训练集与开发集。七种必要结构必须在监督信息和实际预处理数组中存在。缺少必要结构的病例保留排除原因，其余合格病例参与 init。正式两支训练需要至少四名训练患者和一名开发患者，不足时明确报告。

独立结果位于 `auto-targets/`、`auto-init64/`、`auto-init128/`、`auto-six-way/`。`auto-target-report.json` 逐例记录种子、心室最近点、物理距离、血管纳入比例与范围。原 `targets/` 继续用于 blood，已有 blood 检查点的监督身份保持一致。

## 原目录续跑

先确认原 Xu 进程已经退出，再更新代码并按照[目录迁移说明](storage-and-mirrors.md)迁移旧过程目录。SAM 任务也须在迁移期间停止。确认迁移完成后使用新工作目录续跑。

```bash
git pull --ff-only https://gh-proxy.com/https://github.com/tianwuasd/project.git main
source /data5/zhougaowei/zhangruichen_workspace/dataset1282_runtime/training-env.sh
W=/data5/zhougaowei/zhangruichen_workspace/dataset1282_work
O="$W/experiments/xu-train-1794048634"
nohup "$PY" -B -u scripts/xu_pipeline.py run \
  --prepared "$W/work/prepared/zmic-source-v030/hu-marginal" --output "$O" \
  --profile paper --supervision marginal --devices all-visible --threads 4 \
  --prefetch --wait-seconds 300 --auto-init --resume >>"$O.log" 2>&1 < /dev/null &
```

自动扩展设置在 `.runtime/auto-init-extension.json` 冻结，此后续跑继续提供 `--auto-init` 和相同半径参数。原 init 跳过记录仍会显示 `Verified`；新增模型进度看 `auto-init64`、`auto-init128`。先完成新增的两支和六路融合，再从原检查点继续 blood，最后执行 blood 融合及血管处理。

已完成的 `with-blood`、`vessels` 或最终结果会阻止在同目录新增这项扩展，避免覆盖已有完整结果。源码差异仅通过配置中已审核的完整摘要映射接受；数据、模型设置、未知源码差异继续拒绝。完整 epoch 检查点继续训练，未提交的半轮需要重算。

## 本地验证

2026-11-11，在 RTX4060 Laptop、PyTorch 2.12.1+cu126 上，用正式 blood 网络和 512×512 七切片合成输入完成 ABBA 对照。每组先暖机，再计三次更新或预测；训练每步中位耗时由 1.0672 秒降至 0.9553 秒，约 1.117 倍；开发集由 0.5100 秒降至 0.2474 秒，约 2.062 倍。对照只覆盖计算部分，未包含完整病例读取、完整轮次或服务器等待。GPU 峰值保留量约 5.03 GiB。

数值检查覆盖七帧分组后的输出、loss、梯度与 Adam 更新；重计算开关检查参数键和梯度；真实压缩文件检查读取、并发、文件变化拒绝和释放。新 init 以 64³、128³ 的轻量短测配置完成真实 GPU 更新、保存恢复及预测，并与旧四路权重、旧 blood 权重完成六路和 blood 融合；156 个原任务文件摘要保持一致。

旧实现生成的真实 encoder 检查点在当前实现继续完成 sequence；原 encoder 权重、优化器检查点和合同的字节摘要保持不变。Xu 回归共 176 项，171 项通过，5 项按前提条件跳过；另有 8 项兼容测试通过。三例本地真实 CT 的自动标签检查有两例合格，一例缺必要结构。正式全队列的 init 人数与范围需由运行时报告确认。

这些验证不评价分割精度，也不代表 RTX3090 或目标 Linux 多卡的耗时验收。
