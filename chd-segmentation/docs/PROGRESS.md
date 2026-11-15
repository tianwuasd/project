# 阶段进度记录

`chd_segmentation.observability.progress.Progress(config, stage, total=None, reused=0, interval_seconds=15, scan_run_dir=False, resource=None)` 由业务调用方创建。调用方通过 `update(...)` 提供真实计数；模块不会扫描病例清单推断进度。支持 `done`、`new`、`reused`、`failures`、`pending`、`active`、`current_case`、`block_done`、`block_total`、`epoch`、`step`、`storage_bytes` 和 `resource`。这些计数为截至当前的累计值，调用方负责保持真实。

`snapshot(force=False)` 按至少 15 秒的间隔写出 `run_dir/progress-{stage}.json`，并向 `events.jsonl` 追加 `progress_snapshot` 事件。`close()` 写出最终快照。JSON 通过现有原子替换方法写入。快照包含时间、耗时、阶段计数、新完成项速率、ETA、资源信息、磁盘剩余容量及调用方登记的本任务存储字节数。

终端输出简洁中文行，同时显示病例完成量和当前阶段块数。例如病例 75/76、当前 ct_1126 图像重采样 24/60 块，剩余病例与剩余块分别展示。结构化进度的 `block_remaining` 以本阶段实际块总数计算；总量未知时显示未知。详细进程列表和 cgroup 明细仅在定期覆盖的 `memory.json` 保留，块日志写资源摘要。

速率使用上次快照以来新增的 `new` 数量除以经过时间；`reused` 不计入计算速度。没有可用总量、已新增工作不足以计算速率或剩余量不确定时，ETA 为 `null`。进程 CPU 使用率使用 `psutil.cpu_percent(interval=None)`，非阻塞读取，每次资源采样间隔至少 15 秒。CUDA 任务使用既有 `nvidia-smi` inventory 查询 GPU；该查询不初始化 CUDA。调用方可通过 `resource` 传入内存保护等现成资源快照。

磁盘剩余空间来自当前任务目录所在磁盘。`storage_bytes` 由调用方累计登记，模块不按每批次遍历目录。`scan_run_dir=True` 时，初始化阶段最多检查 1000 个文件或 100 毫秒，并用 `run_dir_scan_partial` 标记结果是否可能不完整；默认不扫描。

任何文件写入、事件写入和资源查询错误都会直接抛出，调用方应让错误继续传播。进度类仅负责记录数据，不负责决定病例完成、训练更新或失败状态。
