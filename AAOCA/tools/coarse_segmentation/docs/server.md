# Linux 一体化服务器面板

本说明是运行方法，不表示已在生产服务器部署或完成真实模型推理。准备 Python 3.11–3.13 和当前仓库代码即可启动面板，无需先安装 NumPy、PyTorch 或 TotalSegmentator。

## 启动与打开

在仓库 `AAOCA/tools/coarse_segmentation` 目录运行：

```bash
python3 server.py --workspace /srv/private/aaoca-seg
```

工作区必须位于源码之外，由当前用户可写。默认后台启动，立即返回终端；地址和访问令牌保存在工作区。重复命令会连接已有实例，不创建第二个面板。不要以 root 身份运行。

VS Code Remote SSH 的“端口”面板添加 `8787`，在本机浏览器打开转发后的地址并输入令牌。也可在本机另开终端：

```bash
ssh -N -L 8787:127.0.0.1:8787 USER@SERVER
```

访问 `http://127.0.0.1:8787`。本地端口冲突时可把转发左侧端口改为 8788。面板只监听服务器回环地址，不需要开放公网端口。工作区和含许可的配置只放私人目录，不上传 GitHub。

## 在面板完成准备

1. **环境安装**：选择“自动适配 NVIDIA 驱动”，点击“安装 / 修复环境”。只创建工作区内的 `venv`，不升级系统驱动或其他研究环境。安装固定 TotalSegmentator 2.18.0、PyTorch 2.7.1 / torchvision 0.22.1。自动方案保守地为驱动 >=560.35 选择 CUDA 12.6、>=520.61 选择 CUDA 11.8，无 NVIDIA 查询结果时选择 CPU；更旧或未知驱动给出提示。手动方案也需安装后的实际 CUDA 检查通过。
2. **许可**：输入官方非商业许可，点击“验证并保存许可”。如果从私有配置导入，留空直接点击即可。成功后删除临时许可输入，只保留 TotalSegmentator 的私有配置；失败保留输入供重试。许可通过官方接口验证，不跳过校验。无需在终端粘贴许可命令。
3. **清单与结果**：填写服务器 CSV 路径，或从电脑导入包含服务器绝对路径的 CSV。CSV 使用 `case_id,image,spacing_confirmed,hu_confirmed` 四列。输入必须有真实 HU/mm 来源依据，不能把归一化到 `[-1,1]` 的影像作为 HU。原影像不移动、不覆盖。
4. **资源**：环境检查后在设备列表选择获准使用的可见 GPU，确认分配并保存。默认 CPU，避免自动抢占显卡。保留启动面板时继承的 `CUDA_VISIBLE_DEVICES`；可见编号和 UUID 一起保存，执行前重新探测，映射变化会阻止启动。改变调度器分配后应重新启动面板并重新选择设备。
5. **试跑**：选前 3 个体积，点击“只预检”。预检通过后开始粗分割；复核结果后再扩大量。首次实际推理会下载相应模型，安装及模型下载依赖服务器网络。客户端断线不等于服务器离线。

页面显示安装、检查、许可和推理的真实状态与日志。仅提供已结束体积数与当前阶段，不估算没有依据的剩余时间。安装和检查期间不可修改设置或启动另一项操作。

## 私有配置导入

可以把个人配置及 `manifest.server.csv` 上传到同一个私人目录后启动：

```bash
python3 server.py --workspace /srv/private/aaoca-seg \
  --profile /srv/private/inputs/server.private.json
```

配置格式如下。实际许可仅写入自己的私人文件；本仓库不提供真实许可。相对清单路径以该 JSON 所在目录为基准。

```json
{
  "manifest": "manifest.server.csv",
  "output_root": "/srv/private/aaoca-results",
  "device": "cpu",
  "threads": 2,
  "torch_build": "auto",
  "gpu_confirmed": false
}
```

可增加字符串字段 `license`。导入只保存许可输入，安装环境后仍须在面板点击验证。已有后台面板时，重复启动不重新导入配置，避免改变现有任务。

## 断线与恢复

- **关闭网页、关闭 SSH**：面板和任务均以独立后台进程运行，标准输入关闭、日志写入文件，通常不受会话断开影响。不必额外使用 tmux/nohup。
- **重连**：重新连接服务器并转发端口，再次打开原地址。忘记令牌可执行 `python3 server.py --workspace /srv/private/aaoca-seg --status`。
- **仅面板退出**：后台任务继续。用同一工作区启动面板后，依据持久任务记录和 PID/启动身份重新显示；可请求停止自己的分割任务。
- **服务器关机、重启、管理员结束进程、调度器回收任务或配置了退出即清理用户进程**：不能保证继续运行。该工具不绕过集群调度或系统会话清理政策，也不自动开机启动。
- **中断后再运行**：已经完整完成且输入、源码、设置、依赖及产物哈希匹配的体积复用；未完成体积创建新 attempt，重新跑三个模型，不是模型内部断点续算。
- **遗留 `.run.lock`**：不自动删除。核对 PID、父子进程、模型日志和原任务是否仍在运行后再人工处理；不能只因为 SSH 断线就删锁或结束其他进程。

“停止本次任务”只给当前持久任务写入停止请求。模型阶段终止该任务自己的模型进程，输入检查或合并阶段等待安全检查点。安装与环境检查请等待完成，页面不强行中断依赖写入。

如需仅关闭面板（后台任务继续）：

```bash
python3 server.py --workspace /srv/private/aaoca-seg --stop-panel
```

生产任务运行时不更新代码或依赖。推理任务提交时会保存清单、设置和源码副本，便于追溯。

## 文件位置

```text
工作区/
  settings.json          # 路径、线程、设备等普通设置
  environment.json       # 最近环境检查，不含许可
  private/panel.token    # 面板访问令牌，Linux 权限 600
  private/totalseg/      # TotalSegmentator 私有许可配置
  venv/                  # 独立 Python 环境
  weights/               # 模型权重缓存
  jobs/<任务ID>/         # 冻结清单、源码、状态、日志、停止请求
  service.log            # 面板启动日志
```

粗分割输出写入页面设置的结果目录，结构见主 README。工作区整体保持私人访问；不要提交环境、日志、清单、权重或结果。Linux 使用内核文件锁避免同一工作区的并发操作；PID 身份包含启动时间及 boot ID，避免误认重用的 PID。

## 环境问题

- 缺 `venv` / `ensurepip`：用包含 venv 的 Python 3.11–3.13 启动面板，可用现有 Miniforge 提供独立 Python；面板不使用 sudo 安装系统组件。
- 安装失败或网络中断：查看面板日志，解决网络/磁盘问题后再点击安装。不会无限重试。服务器不能直接复用 Windows 本机的 `127.0.0.1:7897` 代理地址。
- `torch.cuda.is_available()` 为 false：核对当前分配、驱动和安装方案；不能从 GPU 型号或空闲显存推断 CUDA 已可运行。
- 未知间距、坐标头冲突、非原始 HU：回查转换依据，不能修改头信息使检查通过。

依据：[PyTorch 固定版本安装](https://pytorch.org/get-started/previous-versions/#v271)、[NVIDIA 驱动兼容](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)、[TotalSegmentator 2.18.0 许可入口](https://github.com/wasserth/TotalSegmentator/blob/v2.18.0/totalsegmentator/bin/totalseg_set_license.py)。实际 GPU/模型兼容性以服务器小批量运行验证为准。
