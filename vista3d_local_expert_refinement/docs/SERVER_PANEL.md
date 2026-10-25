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
| e | 独立环境准备/核验；无需数据或 GPU；不会覆盖计算任务偏好 |
| s / t | 单次任务概览 / 每5秒持续查看安装、步骤、病例与模型进度 |
| m | 正式模型 epoch、train/val loss 与实测轮次 ETA；可持续刷新 |
| l | 日志中心：总日志、步骤、单模型、训练/预测队列；单次尾读或动态跟随 |
| p | 本任务进程树（角色、CPU累计、RAM、运行时间）或本用户全部进程 |
| g | 每秒显示一次显卡；Ctrl+C 返回，不影响后台任务 |
| x | 只向指定任务身份匹配的进程发停止信号，不按显卡杀其他人进程 |
| u | 从认证 GitHub 原站获取本项目，备份并选择独立发布目录 |
| b / q / 0 | 子菜单返回／退出；退出面板不停止后台任务 |

v0.2.0 按 CT 母版重整了查询导航和环境过程。`h` 可选任一历史任务进入它自己的概览/模型/日志/进程/结果子菜单，不改变最近任务。运行中隐藏新建/续跑入口并解释原因；`x` 仍需选定具体任务和确认停止。输入页提示 `b/q`，新建完成后用中文预览名称、目录、顺序及获准 GPU，再输入 RUN。

## 先准备环境、再选择计算

打开面板后选 `e` → `1` → 核对目录 → `RUN`。程序优先复用可用 Conda 程序，计算环境仍独立创建在当前 runtime；缺少 Conda 时下载并校验固定 Miniforge。查询本身无需 Conda、PyTorch 或 nnU-Net。系统入口需 Python 3.11+；默认 python3 版本较旧时：

```bash
VISTA_CONTROL_PYTHON=python3.11 bash start_server.sh
```

安装按“官方安装包下载→SHA256校验→独立 Python→固定 PyTorch→nnU-Net→版本与依赖核验”落盘显示，复用时跳过相应安装阶段；下载显示实际收到的字节，依赖解析/解压没有可可靠计算的百分比。`s/t` 查看阶段，`l` → 总日志 → 持续查看可看完整工具输出。安装记录 `任务/environment.json`，总日志 `任务/launcher.log`。环境成功凭证绑定依赖配置摘要，旧空凭证会重新核验并安装固定依赖；失败不生成新成功凭证。pip 的目标目录配置被隔离，不改系统环境或其它研究环境。

环境任务完成后回 `n` 创建计算任务，不要把仅安装环境的任务当作预处理输入。已有数据清单的历史任务才出现在输入列表中；环境和缓存可以独立复用。旧任务续跑仍使用旧源码快照，无法自动获得新版 worker 的进度字段；新建/独立重跑才使用新快照。

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
bash start_server.sh --query logs --log-source 0 --follow
bash start_server.sh --query logs --log-source step:train --follow
bash start_server.sh --query status --follow
bash start_server.sh --query processes
bash start_server.sh --query processes --process-scope all
bash start_server.sh --query gpu --follow
bash start_server.sh --query results --result-section cases --result-detail detailed
bash start_server.sh --resume-job latest --resume-check full
bash start_server.sh --stop --job "/实际结果目录/任务编号"
```

可用 `--job` 查指定旧任务。查看历史不会改变最近任务。simple 检查小文件摘要、大文件大小和修改时间；full 检查全部摘要；none 只跳过历史产物检查，源码与计划完整性仍检查。恢复使用任务原快照，更新程序不会替换旧任务代码。只有阶段实际结束才标记 completed，提交成功不等于训练成功。

`t` 显示按步骤数量统计的完成数，不能当作按计算量加权的总百分比。预处理另显示导出、指纹、规划、逐病例等阶段及已保存的来源病例数，正式训练读取实际 epoch/loss，预测只数完整病例记录。过期队列标为历史记录；没有有效计时就不估 ETA，等待显卡不代表模型在训练。日志首次最多读取每文件约12KB尾部，持续模式只追加有界新内容、发现后生成的模型文件并处理截断；约10秒附带一次状态，Ctrl+C 返回。排队和后台进程存在本身不证明计算已推进。

停止时用 Linux boot ID、PID 与进程启动时间共同核对，避免 PID 重用。保留最后完整 checkpoint；子进程若未退出仍保有锁，拒绝另开写任务。不要手工删除锁来绕过活动进程。

## 历史输入与独立更新

`--input-job "/旧任务目录"` 可复用兼容来源缓存；工作区采用独立文件复制，不用可互相改写的硬链接。工程 ID、数据路径、协议摘要、实现、规划与选择范围不兼容时拒绝或在包含 prepare 的新任务里建立新缓存。训练输入不能混用旧全标注工程；单独预测必须有正式完成模型。原输入始终保留。

更新入口 `bash start_server.sh --update` 检查已核实的 `tianwuasd/project` origin、当前项目未提交修改和快进关系，在 runtime 下备份和发布，只切换本项目面板；原检出和旧任务快照保留。**不使用公共镜像**。服务器没有私有仓库凭据时更新会报认证错误并保留当前版本，需先在服务器配置 GitHub 访问。

在服务器上进入已有 `project/vista3d_local_expert_refinement` 后更新并重开：

```bash
bash start_server.sh --update
bash start_server.sh
```

如果原来用过 `--runtime /某目录`，两条命令都加相同 `--runtime /某目录`，否则会进入另一个默认运行空间。活动任务会阻止更新；从面板选明确目标停止并等实际退出后再更新。受管更新只切换本项目的独立发布目录，不需要直接在共享 monorepo 执行 pull。服务器代理与 Windows 本机代理是两台机器的设置，本项目继承服务器现有网络与 GitHub 认证，不替用户写 7897。

CT 母版适配及未纳入的产物上传/跨阶段持卡等边界见 [设计对照表](superpowers/specs/2026-10-25-panel-template-design.md)，本轮测试证据见 [面板验证](PANEL_TEMPLATE_VALIDATION.md)。

GPU 监控只是实时探测，不是集群资源预约。必须遵守已有调度分配；程序与 `CUDA_VISIBLE_DEVICES`／SLURM 范围取交集，繁忙时等待，不抢占外部进程。显存等待阈值至少 6000 MiB，并随规划预算提高；OOM 时保留日志和检查点，不自动改 batch size 后混入原实验。

结果目录含 `plan.json`、`state.json`、`code/`、`logs/`、`receipts/`、`workspace/`。正式模型在 `workspace/07_experiments/formal/`，预测在 `workspace/08_results/predictions/`，最终 CSV/报告在 `workspace/09_reports/final_evaluation/`。
