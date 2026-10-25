# wholeheart 基线服务器面板

按 `server-engineering` 技能的命名任务、冻结源码、历史／续跑、只读查询、明确 GPU 范围和独立更新要求实现。入口是终端中文菜单，不需要 Web 服务。查询只依赖 Python 标准库；计算阶段使用独立 Conda 环境。Linux x86_64、Python 3.11+、Bash、Git、flock 和 NVIDIA 驱动为服务器前提。安装器不改驱动、不用 sudo。

## 获取代码并打开

仓库在本次核实时为私有。使用服务器已配置的 GitHub SSH 或 HTTPS 凭据；不将 token 写进命令、计划或日志。可在已有检出中进入该项目，或：

```bash
git clone --filter=blob:none --sparse git@github.com:tianwuasd/project.git
cd project
git sparse-checkout set vista3d_local_expert_refinement
cd vista3d_local_expert_refinement
bash start_server.sh
```

新建任务时填写名字和真实数据目录。默认运行环境位于 `~/.local/share/vista3d_baseline_runtime`，结果位于 `~/vista3d_baseline_results/时间戳_随机编号/`；二者以及原始数据目录必须相互独立。未设定任何旧用户服务器绝对路径。没有复制数据到 GitHub。

| 操作 | 行为 |
|---|---|
| n 新建 | 名称、步骤、数据、输出、B0、留出方向、规划预算、获准 GPU |
| r 续跑／追加 | 使用原始源码快照和冻结设置；恢复未完成步骤或追加后续步骤 |
| k 独立重跑 | 新目录、新缓存，保留原任务结果 |
| h / o | 历史／结果；摘要、病例与七结构、训练、文件，支持简洁/详细和分页 |
| s / m / l / p | 状态、模型、日志、本用户进程查询 |
| g | 每秒显示一次显卡；Ctrl+C 返回，不影响后台任务 |
| x | 只向指定任务身份匹配的进程发停止信号，不按显卡杀其他人进程 |
| u | 从认证 GitHub 原站获取本项目，备份并选择独立发布目录 |
| b / q / 0 | 子菜单返回／退出；退出面板不停止后台任务 |

## 建议先短测

下面的 `/实际路径/Wholeheart_Train_Dataset` 必须替换；GPU 0 只是示例，需填你获准的编号。默认只跑 A+B→G。

```bash
bash start_server.sh --name "B0_G_首次短测" \
  --data-dir "/实际路径/Wholeheart_Train_Dataset" \
  --tasks environment,prepare,smoke --models B0 --scope ct-first \
  --gpus 0 --gpu-count 1 --gpu-memory 5 --non-interactive
```

首次后台安装固定 Miniforge 与 Python 环境、PyTorch 2.8.0+cu128／torchvision 0.23.0+cu128／nnunetv2 2.8.1。安装器 SHA256 已对官方发布核对。CPU 安装与预处理先执行，只有 GPU 阶段才等待获准卡。一次命名任务独占其 runtime 写锁；三个折可在该任务内部最多三张获准卡并行，每模型单卡，不是分布式训练。

完成短测后查看结果，再追加训练；这是实际启动长训练的命令：

```bash
bash start_server.sh --resume-job latest --append-tasks train
```

完成正式训练后追加目标评价和汇总：

```bash
bash start_server.sh --resume-job latest --append-tasks predict,summarize
```

也可在新建时选 `environment,prepare,smoke,train,predict,summarize`，确认后自动按阶段执行。运行 `--scope all` 时独立训练三个留出方向。CPU-only 任务没有选过 GPU，不能直接追加 GPU 计算；请新建任务、选其历史输入并明确 GPU。

## 查询、恢复与停止

```bash
bash start_server.sh --query history
bash start_server.sh --query status
bash start_server.sh --query models
bash start_server.sh --query logs
bash start_server.sh --query gpu --follow
bash start_server.sh --query results --result-section cases --result-detail detailed
bash start_server.sh --resume-job latest --resume-check full
bash start_server.sh --stop --job "/实际结果目录/任务编号"
```

可用 `--job` 查指定旧任务。查看历史不会改变最近任务。simple 检查小文件摘要、大文件大小和修改时间；full 检查全部摘要；none 只跳过历史产物检查，源码与计划完整性仍检查。恢复使用任务原快照，更新程序不会替换旧任务代码。只有阶段实际结束才标记 completed，提交成功不等于训练成功。

停止时用 Linux boot ID、PID 与进程启动时间共同核对，避免 PID 重用。保留最后完整 checkpoint；子进程若未退出仍保有锁，拒绝另开写任务。不要手工删除锁来绕过活动进程。

## 历史输入与独立更新

`--input-job "/旧任务目录"` 可复用兼容来源缓存；工作区采用独立文件复制，不用可互相改写的硬链接。工程 ID、数据路径、协议摘要、实现、规划与选择范围不兼容时拒绝或在包含 prepare 的新任务里建立新缓存。训练输入不能混用旧全标注工程；单独预测必须有正式完成模型。原输入始终保留。

更新入口 `bash start_server.sh --update` 检查已核实的 `tianwuasd/project` origin、当前项目未提交修改和快进关系，在 runtime 下备份和发布，只切换本项目面板；原检出和旧任务快照保留。**不使用公共镜像**。服务器没有私有仓库凭据时更新会报认证错误并保留当前版本，需先在服务器配置 GitHub 访问。

GPU 监控只是实时探测，不是集群资源预约。必须遵守已有调度分配；程序与 `CUDA_VISIBLE_DEVICES`／SLURM 范围取交集，繁忙时等待，不抢占外部进程。显存等待阈值至少 6000 MiB，并随规划预算提高；OOM 时保留日志和检查点，不自动改 batch size 后混入原实验。

结果目录含 `plan.json`、`state.json`、`code/`、`logs/`、`receipts/`、`workspace/`。正式模型在 `workspace/07_experiments/formal/`，预测在 `workspace/08_results/predictions/`，最终 CSV/报告在 `workspace/09_reports/final_evaluation/`。
