# Linux GPU 服务器操作

本页面提供运行步骤，不表示已经连接或部署到某台服务器。用自己的代码、数据和授权 GPU 路径替换示例；先完成单例/小批量复核。

## 1. 使用独立环境

进入 GitHub `project` 仓库的 `AAOCA/tools/coarse_segmentation`，不要混用资料处理 GUI 的环境：

```bash
cd /path/to/project/AAOCA/tools/coarse_segmentation
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python run.py doctor
```

如果仓库使用 sparse checkout，确保保留 `AAOCA`。代码更新后不要覆盖正在运行的实验；先结束当前任务。根据服务器驱动选择 PyTorch CUDA 构建，安装后用以下命令只读检查：

```bash
nvidia-smi
printf '%s\n' "$CUDA_VISIBLE_DEVICES"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

`gpu:0` 指当前进程可见的第 0 张 GPU，不一定是物理 0 号。保留调度器已有的 `CUDA_VISIBLE_DEVICES`，不要清空限制。只使用明确分配的 GPU；面板的空闲信息不等同于资源预约。当前为逐例串行推理，不包含多卡并发调度。

## 2. 配置模型许可及私人缓存

到 [官方许可页](https://backend.totalsegmentator.com/license-academic/)申请免费非商业许可。可以选择用户可写的缓存目录，然后按官方入口设置：

```bash
export TOTALSEG_HOME_DIR=/srv/private/preseg/totalseg_config
export TOTALSEG_WEIGHTS_PATH=/srv/private/preseg/totalseg_weights
mkdir -p "$TOTALSEG_HOME_DIR" "$TOTALSEG_WEIGHTS_PATH"
totalseg_set_license -l YOUR_LICENSE
python run.py doctor
```

替换密钥时避免将真实密钥存到脚本或版本库。配置与面板必须由同一账号读取。安装和下载需网络；不建议在已有生产研究环境直接升级依赖。

## 3. 清单与预检

从已有研究清单导出 AAOCA，服务器研究目录应保留 `AAOCA/` 之后的结构：

```bash
python run.py inventory --inventory /srv/private/research/CT总清单.json \
  --data-root /srv/private/research --manifest /srv/private/preseg/aaoca.csv
python run.py plan --manifest /srv/private/preseg/aaoca.csv \
  --output-root /srv/private/preseg/results --limit 3
```

没有研究总清单时按示例 CSV 手工登记，确认输入是原始 HU。不要把另一个预处理模块输出的 `[-1,1]` 数据当作 HU，不要使用真实间距未知的 CHD 数据来代替 AAOCA。

## 4. 保持独立面板运行

推荐在 `tmux` 会话里启动；服务器若没有 tmux，可由管理员提供等价的会话保持方式。

```bash
tmux new -s aaoca-preseg
source /path/to/project/AAOCA/tools/coarse_segmentation/.venv/bin/activate
cd /path/to/project/AAOCA/tools/coarse_segmentation
python run.py panel --manifest /srv/private/preseg/aaoca.csv \
  --output-root /srv/private/preseg/results --device gpu:0 --threads 2 --port 8787
```

记下当前终端显示的令牌。按 `Ctrl+B` 再按 `D` 离开 tmux 而保持面板运行；`tmux attach -t aaoca-preseg` 返回。令牌仅保存在面板进程内存，重启后更换。

在自己的 Windows/macOS/Linux 电脑另开终端：

```bash
ssh -N -L 8787:127.0.0.1:8787 USER@SERVER
```

浏览器访问 `http://127.0.0.1:8787`，输入令牌。若本地 8787 已占用，可改成本地 8788：`ssh -N -L 8788:127.0.0.1:8787 USER@SERVER`，浏览器访问 8788。无需把服务器 8787 端口对外开放。

## 5. 操作与恢复

1. “选前 3 例” → “只预检”，查看几何与确认提示。
2. 查看已分配的 GPU、可见范围和许可状态，再点击“开始粗分割”。首次会下载所需模型。
3. 查看体积计数、阶段和模型日志。当前没有可靠的模型内部百分比或预计时间。
4. “停止本次任务”只设置本面板任务的停止请求，模型阶段会终止自有子进程；输入校验和标签合并可能要到下一安全检查点。
5. 再次开始时，完整结果逐文件校验后复用；失败/未完成病例使用新的 attempt。不是任务内断点续算。
6. 通过 Slicer 审核结果后，才扩大批量或用于训练。合并标签不直接覆盖人工标签。

关闭网页不会停止任务。退出面板前先点击停止并等状态结束；不要依靠关闭 SSH 来中断模型。如果面板重启而旧任务仍在运行，新面板不接管或杀死旧 PID。回到原终端处理；若机器断电留下 `.run.lock`，先查看其中 PID 并核对进程身份和子进程，确认已退出后再人工处理锁。程序不会自动删除遗留锁。

只看状态可以读取输出根目录的 `status.json`；模型日志在 `cases/<代号>/attempt-*/<task>.log`。`latest.json` 始终指向上一次完整成功记录；新失败记录在新 attempt 中，不覆盖旧成功结果。

## 常见问题

- **许可未配置**：`doctor` 检查使用相同的 `TOTALSEG_HOME_DIR`；实际有效性由 TotalSegmentator 验证。
- **显存不足**：停止当前任务后核对分配和进程占用；可用 CPU 检验流程，或换获准使用的更大显存 GPU。不要为了冠脉直接启用低分辨率 fast。
- **未知空间单位 / 坐标冲突**：回查 DICOM 与转换记录；不能修改头信息使检查通过。
- **空冠脉、断裂或相邻结构粘连**：保留 QC 和原二值结果，逐例人工修订。本工具不自动以正常解剖假设补连接。
- **磁盘增长**：每次失败保留新 attempt，合并图与二值图同时保存。人工确认不再需要某次结果后自行归档；程序没有自动清理按钮。
