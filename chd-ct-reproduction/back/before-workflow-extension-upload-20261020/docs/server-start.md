# zmic44：按步骤运行 CHD

服务器默认采用 wholeheart 的独立后台运行方式：提交后可以断开 SSH，任务继续运行并保存日志。菜单支持多选，预处理、训练、预测、评估和诊断仍可分别启动。只有所选步骤使用 GPU 时才询问显卡；模型配置只在训练步骤出现。CHD 保留自己的六阶段模型、七结构标签及配置。

## 先改哪一个文件

集中设置在 `configs/servers/zmic44.json`。第一次也可以直接运行 `bash start_server.sh`，按中文提示设置；回车选择默认值。

| 设置 | 默认值 |
|---|---|
| `dataset` 原始数据 | `/data_nas/zhangruichen/baidu_import/ImageCHD_dataset` |
| `prepared` 缓存 | `/data_nas/zhangruichen/chd_ct_data/imagechd7` |
| `results` 模型、日志、预测、评估 | `/data_nas/zhangruichen/chd_ct_results` |
| `runtime` 环境与下载缓存 | `/data5/zhougaowei/zhangruichen_workspace/chd_ct_runtime` |
| `config` 模型设置 | `configs/chd.yaml` |
| `gpu_count` 训练最大并发卡数 | `1`，可设 1—5；不足时少卡运行，无卡时等待 |
| `gpu` 候选卡范围 | `auto`；也可写 `0,2`，候选卡数量不必等于 gpu_count |

原始数据默认路径是本项目约定位置，尚未确认已上传；数据必须已解压，符合 ImageCHD 的 image/label 配对规则。不能把 Wholeheart_Train_Dataset 直接当 ImageCHD 输入。路径不正确时用 `--dataset /实际路径/ImageCHD_dataset` 覆盖。

设置优先级：**命令行 > runtime 中记住的设置 > 服务器配置文件**。保存在 `<runtime>/.server-settings.json`，每一步另存只含该步骤有效参数的 `server-settings.json` 快照。更换配置默认值后，用 `--reset-settings` 忽略上次选择；不删除缓存或模型。模型结构和 gate 开关仍在 CHD YAML 中。

## 菜单多选，默认后台运行

在服务器的 `project/chd-ct-reproduction` 中执行：

```bash
bash start_server.sh
```

0.9.0 起，先进入任务面板。确认 runtime 后自动查看最近任务的进程身份；有任务运行时可以查询或停止，没有任务运行时可以新建或选择历史任务续跑。面板按回车默认查询状态，不会直接启动。

| 面板输入 | 功能 |
|---|---|
| `s` / `g` / `p` | 流程状态（含本流程保留的 GPU）/ 显卡状态（每 1 秒刷新，Ctrl+C 返回）/ 本用户的工程进程 |
| `m` | 六个模型的已完成 epoch、目标轮数、train/val loss、学习率及 GPU UUID |
| `l` | 选择步骤和模型查看详细日志；可看一次或持续刷新 |
| `x` | 中断当前任务并保留检查点；先显示目标名称和目录，再确认 |
| `u` | 更新程序，多个 GitHub 镜像自动切换；运行中先确认中断并等待退出 |
| `h` | 选择历史任务，查询其状态和日志 |
| `n` | 无任务运行时新建，输入本次任务名称 |
| `r` | 无任务运行时，从历史列表选择中断/失败任务续跑 |
| `0` / `q` | 退出面板，后台任务继续；各输入页也可用 `q` 退出 |
| `b` | 取消当前输入并返回面板；Ctrl+C 也可取消输入/退出日志查看 |

新建后输入 `0,1,2,4,5`，即环境检查 → 预处理 → 软件短测 → 正式训练 → 测试；逗号、空格或中文逗号均可。已有预处理缓存时可选 `0,2,4,5`；保留预处理步骤则校验并复用/接续缓存。下表是**新建任务后的步骤菜单**，与上面的任务管理面板分开，桌面 `start.py` 仍为单选。步骤编号 `0` 和 GPU 编号 `0` 保留原意，不代表退出。

名称可以包含中文和空格（1—80 个字符），只用于显示；底层目录仍为时间戳＋随机 ID，重名不会覆盖。续跑保留原名称、配置、目录和检查点；旧版无名称记录显示目录名。已全部完成的任务可查询，不重复续跑。历史列表包含同一 runtime 已登记的任务和结果目录中发现的旧任务；旧任务在其他目录时用 `--results /旧结果根目录` 查看，或 `--resume-job /完整workflow目录` 直接指定。

| 编号 | 步骤 | 编号 | 步骤 |
|---|---|---|---|
| 0 | 环境检查 | 8 | 导入诊断标签 |
| 1 | 预处理 | 9 | 提取诊断特征 |
| 2 | 软件短测 | 10 | 训练诊断树 |
| 3 | 正式尺寸预检 | 11 | 诊断与解释 |
| 4 | 正式训练 | 12 | 诊断评估 |
| 5 | 测试集预测与评估 | 13 | 合成分割测试 |
| 6 | 独立预测（默认全部病例） | 14 | 合成诊断测试 |
| 7 | 独立分割评估 | | |

程序按表中顺序执行所选步骤，去重，不自动添加未选步骤。提交前显示输入、输出和执行顺序；缺少必要输入时不提交任务，交互模式返回面板。同一次流程中，后续步骤自动接用前面生成的结果，例如测试使用本次正式训练的模型。显式传入的 `--models`、`--features` 等优先。正式训练自身始终先做真实尺寸预检。

非交互运行同一流程（例子假定已获准使用 GPU 0、1）：

```bash
bash start_server.sh --tasks environment,preprocess,smoke,train,test \
  --name '首次训练' --gpu-count 2 --gpu 0,1 --non-interactive
```

已有缓存时可去掉 `preprocess`，也可保留它进行来源校验；更换数据来源加 `--dataset /实际路径/ImageCHD_dataset`。0.11.0 起在预处理缓存目录后选择策略：

```text
预处理缓存策略：1 尝试沿用（失败自动新建） / 2 新建 [1]：
```

选 1 先校验旧缓存，完整缓存沿用，未完成缓存接续；来源、参数不一致或缓存损坏/无法识别时，显示原因，保留旧目录并自动新建同级的“原名称_时间_随机ID”目录。选 2 直接生成独立新路径。新路径、改换原因会写入计划和状态，后续所选步骤统一使用最终缓存；上次实际使用的目录会成为下次默认值。表单取消不会创建这些目录。

自动回退最多重试一次；磁盘不足、权限、原始 CT/标签错误、环境错误或用户中断不会被当成缓存不兼容。已有训练/预测产物的流程不能换绑缓存以续训，需新建任务。旧的预处理失败流程若后续步骤尚未启动，更新后用 `r` 续跑也可自动回退。选择 `r` 仍沿用冻结设置和最终路径，不重新弹出新建策略。

**预处理只用 CPU。** 所选流程包含预处理时，前置 environment 也按 CPU 检查，不申请或等待 GPU；到 smoke/preflight/train/predict/test 的实际 GPU 步骤才选卡并等待。新建时填写的 GPU 上限用于后续 GPU 步骤，不构成 CPU 预处理的运行条件。

仅预处理不需要显卡参数，非交互默认尝试沿用，也可显式选择：

```bash
bash preprocess_server.sh --non-interactive
bash preprocess_server.sh --preprocess-cache new --non-interactive
```

提交成功会打印 PID、日志和状态命令。**提交成功不等于训练完成**。默认无需 tmux 或 nohup；关闭本地电脑不影响服务器进程，服务器重启、进程被终止仍会中断。重新登录后：

```bash
bash start_server.sh --status
# 使用了自定义 runtime 时，查看同一个目录：
# bash start_server.sh --status --runtime /个人路径/chd_ct_runtime
# 实时日志使用提交时显示的完整路径：
# tail -f /完整结果路径/某次_workflow/launcher.log
```

一个 runtime 同时只允许运行一个流程，重复提交会被拒绝。任一步失败则停止后续步骤；`pending` 表示未执行，`passed` 才表示完成，`stale` 表示进程已退出或主机重启但没有正常写入结束记录。不要手动删除运行中任务的锁文件。可给启动命令加 `--foreground` 在前台查看日志并等待退出码；此模式下 Ctrl+C 会终止本次流程。

## 更新程序与复用缓存

以上新增操作从 0.10.0 起可用；旧服务器须先通过 Git 更新一次，之后可直接在面板输入 `u`。只退出面板不会中断后台训练；要中断现有任务，选 `x`，确认具体名称与目录，待进程退出后可用 `r` 选择历史任务续跑。中断保留已写入的完整检查点，尚未完成的当前轮可能重跑。

`u` 先检查发布仓库 `tianwuasd/project` 的 `main` 分支和工作区。有本地修改、未跟踪文件、未完成的 Git 操作或分叉时拒绝覆盖，不执行 reset/stash。有任务时确认中断目标，再等最多 60 秒取得同一 runtime 的锁；父进程或训练子进程未退出就取消更新。使用共享代码的其他 runtime 任务也应先停止。

0.11.1 起同时识别 `https://gh-proxy.com/https://github.com/tianwuasd/project.git`，兼容省略 `.git` 和末尾 `/` 的写法。仓库地址识别与下载线路分开，认可这个历史镜像地址后仍走下面的固定线路，不修改 origin；其他仓库或未知镜像仍需核对来源。若旧面板还报“请先核对 origin”，先用 `git remote get-url origin` 确认确实指向本项目，再执行 `git remote set-url origin https://github.com/tianwuasd/project.git`，重新打开面板选 `u` 完成这一次更新。

更新线路依次为 `gh-proxy.org`、`ghfast.top`、`gitclone.com`，最后 GitHub 原站。GitHub 版本查询限时 12 秒，每条下载线路限时 45 秒，失败自动切换。能取得 GitHub main 时要求下载提交一致；原站不可用则比较可访问镜像的提交历史，明确提示无法确认镜像是否最新。不能保证所有线路故障时仍更新成功，也不会把旧镜像直接当作最新版本。沿用服务器自身网络设置，不注入 Windows 本机代理，不向公共镜像传递 Git 凭据。

快进前将当前提交的 CHD 代码保存到 `<runtime>/updates/<时间_编号>/code-before-update.tar`，同目录 `update.json` 记录旧版本、目标版本和各线路结果；数据、缓存和训练结果仍在原目录。成功后面板退出，重新运行 `bash start_server.sh` 加载新版；下载或应用阶段异常也会退出旧面板并提示检查记录。更新不会自动恢复训练。

选中预处理时，按上文 0.11.0 缓存策略选择沿用或新建。执行时核验原始文件、Excel、划分、重采样参数；完整缓存通过后复用，不完整缓存接续缺失部分。旧缓存校验失败时明确记录原因，保留旧内容并自动改用新目录，后续参数仍使用本次计划中的科学设置。

## 随时查询模型进度和详细日志

在工程目录里直接运行。查询只需系统 Python，不安装训练环境、不占训练锁、不触发选卡或新任务；使用自定义 runtime 时，追加同一个 `--runtime /完整路径`。

```bash
# 当前流程、显卡和模型训练记录
bash start_server.sh --query status
bash start_server.sh --query gpu
# 显卡每 1 秒刷新；Ctrl+C 只退出查看，后台训练继续
bash start_server.sh --query gpu --follow
bash start_server.sh --query models
# 每 3 秒刷新模型摘要；Ctrl+C 退出查看
bash start_server.sh --query models --follow
# 自动找到当前步骤的各模型日志；后启动的模型日志也会被发现
bash start_server.sh --query logs --follow
# 指定正式训练中的 all128，只显示末尾 80 行并持续刷新
bash start_server.sh --query logs --step train --model all128 --lines 80 --follow
# 软件短测、总流程日志、步骤调度日志
bash start_server.sh --query logs --step smoke
bash start_server.sh --query logs --step launcher
bash start_server.sh --query logs --step train --model server
# 本用户进程和历史任务
bash start_server.sh --query processes
bash start_server.sh --query history
```

查询历史任务可在 `--query models/logs/status` 后加 `--job-dir /完整workflow目录`，不会改变最近任务指针。步骤用名称定位，不假定训练一定在 `05_train`。独立 preflight 输出在 `train/`，正式训练内部预检在 `preflight/`，入口会识别当前阶段。

三折训练时，模型查询先显示每折状态与 train/val/test 数量，再列出各折六阶段 epoch/loss。0.12.0 的 GPU 训练显示“跨折共享队列”、所有当前运行折及总 GPU 用量；旧串行记录仍显示当前折及其队列。日志查询自动发现 `fold_0` 至 `fold_2` 的已有和后续日志，`--model all128` 可同时跟随三折对应模型。

### 三折共享 GPU 队列（0.12.0）

所有未完成折的模型任务进入同一个队列，优先派发较早折的就绪模型；该折只剩依赖等待时，空闲卡可以领取后续折的独立模型。例如 GPU 7 运行 `fold_0/blood2d` 时，GPU 5、6 可开始 `fold_1/crop64`、`fold_1/crop128`，无需等待整个 `fold_0` 结束。

`--gpu-count 3` 仍表示整个流程最多同时使用 3 张卡，每卡一个模型；不会变成每折 3 张。每折 `blood_lstm` 只在本折 `blood2d` 成功完成后进入运行，不跨折共用编码器。数据划分、科学配置、epoch、随机种子和模型文件格式不变；CPU 或未指定候选 GPU 队列的直接训练仍按原方式执行。末尾只剩少量任务或所有就绪任务受依赖/资源限制时，仍可能有卡空闲。

训练根目录的 `queue.json` 记录全局资源与 `fold_i/模型` 状态，`fold_i/queue.json` 是该折状态，模型日志仍在 `fold_i/stage_logs/`。完整折经过哈希和配置核验后跳过，未完成阶段从完整 epoch 检查点恢复；所有 worker 继承总输出及各折锁，避免父进程退出后重复启动。

旧流程不会自动切换调度器。通过面板 `u` 确认中断具体任务并更新，等旧进程退出后重新打开面板，再用 `r` 选择原任务续跑。更新与恢复不要求重做兼容缓存或已完成折；不能手工覆盖正在运行的代码目录。

日志中的 loss 在每轮训练及验证都完成后才更新。0.13.0 的详细日志查看每 10 秒显示进度摘要：训练读取既有 epoch/timing，非训练阶段读取完成病例数，显示已用时间和估算剩余计算时间；无样本、失败/中断或未知 GPU 等待时显示无法估算。ETA 不包含未来排队，也不是可靠完工时间。队列心跳只表示调度器还在运行；`blood_lstm` 显示依赖等待时需要先完成 `blood2d`。恢复前的旧队列会明确标注，不能据其判断当前进度。无模型独立日志时暂时显示步骤 `server.log`，持续查看会发现后来生成的模型日志。首次日志读取最多取每个文件末尾 256 KiB，避免大日志拖慢查询。

查询和面板中的 Ctrl+C 只退出查看/返回面板；`--foreground` 下 Ctrl+C 才会中断任务。停止任务请用面板 `x` 或 `--stop`，停止请求后再次查状态，确认进程退出再续跑。

## 每一步仍可单独运行

下面是独立入口示例，**按需选择一条提交，完成后再提交下一条**；要连续完成多步，请使用上面的 `--tasks`。

```bash
bash smoke_server.sh --gpu-count 2 --gpu 0,1 --non-interactive
bash train_server.sh --mode train --gpu-count 2 --gpu 0,1 --non-interactive
bash test_server.sh --models /完整路径/某次正式训练/train --non-interactive
bash predict_server.sh --models /完整路径/某次正式训练/train --split auto --non-interactive
bash evaluate_server.sh --predictions /完整路径/某次预测/predict --split auto --non-interactive
bash train_server.sh --mode check --non-interactive
```

`smoke_server.sh` 使用缩小网络，每阶段一个训练 batch 和一个验证 batch；`test_server.sh` 使用已有模型做测试集推理和 Dice 评估。只有短测权重时，明确加 `--allow-smoke`，结果仅验证软件。`train_server.sh` 未指定模式时默认短测；多选菜单中的“4 正式训练”及 `--tasks train` 默认正式训练，也可显式传 `--mode` 覆盖。

训练读取已有预处理缓存，并可生成独立的训练输入加速缓存；测试只读已有模型。无标签 CT 使用独立预测，不能做需要真值的测试/评估。服务器 0.11.0 按上述缓存策略处理已有目录；直接调用底层 `python -m chd_ct.imagechd.preprocess` 仍严格校验 `--resume`，不私自改变输出路径。正式训练失败不会将不完整模型记为上次成功模型。

## CPU 自动配置与小样本测试

默认 `configs/chd.yaml` 在现有 smoke / preflight 开始时自动选取 CPU 加载配置；直接正式训练也会先选取。沿用 `bash start_server.sh` 的小样本测试和正式训练步骤，无需新的测试命令。显式使用 `chd-smoke.yaml` 的缩小软件测试仍保持轻量配置，不能拿它的推荐推断论文输入规模性能。

- 初始参考为每个加载池最多 4 个 worker、每 worker 1 个 Torch 线程、预取 2 个 batch、持久 worker。先按当前 CPU 亲和性/cgroup 配额、可用内存、`/dev/shm` 和任务并发上限收缩；预算包含训练和验证两个加载池。不会照搬整机 256 线程。
- 只用训练划分前两例，三折任务只用首折训练病例。3D 取最大输入尺寸作为代表，2D 和 LSTM 分开比较，使用源配置中的正式尺寸和 batch，先完成缓存准备，再比较至多三个 worker 数。记录两轮热加载的每样本耗时；相差不超过 10% 时优先少用进程。
- 默认候选测量总预算 180 秒、单个候选最多 45 秒；冷缓存构建及校验单独计时，不计入这个测量预算。失败/超时保留 `trial.log`，候选全部未通过时回退 0 worker 并明确记录。该测试测量 **CPU 数据供应**，不使用候选 GPU；随后原有 preflight 继续检查真实模型的 GPU 训练/验证路径。
- 同一数据、源配置、训练病例、主机、CPU 配额、并发上限及 Torch 版本复用推荐；新任务复用前重新检查内存预算。每次输出根目录保存 `cpu-tuning.json`，最终值进入 `effective-config.json` 和训练检查点契约。续训沿用冻结值，不重新调参。
- `.training-inputs/` 默认位于 prepared 目录下，保存带来源 SHA256 和格式版本的 `.npy` 文件；不改变原 NPZ、Excel 重采样或患者划分。内存映射让多个进程共享操作系统文件缓存，避免每个切片重复解压整例、重新计算三维血池标签。首次生成较慢，后续阶段/折复用。默认磁盘预算 32 GiB，保留 2 GiB 余量；这是磁盘预算，不是 RAM 硬上限。不自动迁移到 `/data5`。

配置中的 `data_loading.cache_dir` 可指定个人缓存目录；`cache_max_gib` / `cache_reserve_gib` 控制磁盘预算。需要固定配置时设置 `auto_tune: false`，用 `workers` 或 `stage_workers` 指定进程数；`cache: none` 可关闭加速缓存。没有 `data_loading` 字段的旧配置保留旧加载方式，不能向旧任务配置中直接补字段后冒充原实验续训。

每阶段保存 `*.timing.json`：加载等待、传输＋增强＋模型计算、每轮总耗时。日志约每 30 秒在 batch 完成后打印已完成 batch/样本数；不在卡住的算子中虚报进度。`history.json` 继续只记录确定性的 epoch/loss 等训练结果。

本地验证包含缓存张量一致性、多进程随机增强和断点续训一致性。CPU 小样本推荐不等于完整训练最佳配置，也不能据此承诺 GPU 利用率或提速倍数；实际收益看服务器预检和正式训练的耗时记录。模型结构、学习率、batch、增强策略及正式训练轮数不参与这次自动调优。修改前文件及哈希保存于 `back/before-cpu-optimization-20261016/`。

## 多卡怎样工作

沿用 whole heart 的**一张卡一个模型任务**方式，不使用 DDP，也不把多张显卡显存合并。

- `crop64`、`crop128`、`all64`、`all128`、`blood2d` 可独立训练，最多五个同时运行。
- 默认三折共 18 个阶段实例。0.12.0 起，GPU 训练使用跨折共享队列，总并发受同一个上限约束；CPU 训练仍逐折执行。各折患者划分独立冻结，外层测试病例不参与该折训练或选模。
- `blood_lstm` 必须等待本次 `blood2d` 成功，使用并冻结其编码器。
- 子进程只可见自己分配的 GPU；开始下一阶段前重新检查该卡。任何阶段失败会停止新派发并结束本队列仍在运行的子进程，保留日志与已经完成的权重。
- 同时运行的数量受 GPU 数和可运行阶段数共同限制，后期可能只有一张卡在工作。
- 预处理、缓存检查和评估用 CPU。融合预测/测试从候选池中选择一张空闲卡。`gpu_count` 是训练并发上限，环境检查至多检查这么多张当前空闲卡，不要求凑齐。
- 每个阶段采用 `seed + 10000 × 折号 + 阶段序号`，折号为 0/1/2；单折沿用 `seed + 阶段序号`。单卡顺序与多进程运行使用一致的阶段种子。

菜单选择显卡前先显示实时物理编号、型号、空闲显存、利用率、计算进程占用和可见范围。执行时重新检查，不将菜单快照当作预约。保持尊重 `CUDA_VISIBLE_DEVICES`；卡上已有计算进程、低于 6000 MiB 空闲显存或利用率高于 10% 的卡暂不派发。不会切到 CPU、放宽门槛或扩展分配范围。空闲检查不是集群调度器预约，共享服务器仍应先确认分配。

### 显卡不足时等待和少卡运行（0.8.0）

0.13.0 按最新要求，服务器流程拿到 GPU 后保留到**所有所选 GPU 工作结束**。测试/预测先生成工作网格中间结果，所有 GPU worker 退出后释放持卡，再执行独立 CPU 恢复原网格、压缩和评估。优先使用已保留的卡；未达到本次卡数上限时仍可补入新空闲卡，累计保留数不超过计划中的 GPU 上限，始终受候选池和 `CUDA_VISIBLE_DEVICES` 约束。CPU-only 流程不申请 GPU；混合流程在首次实际 GPU 工作前开始保留，不会在前置 CPU 预处理时占卡。

保留进程仅维持一个很小的 CUDA 张量及上下文，不运行空计算；上下文本身仍占少量显存。自己的调度仅忽略经 PID、boot/start 身份核验的本流程保留进程，其他计算进程仍视为占用，不跳过显存和利用率检查。这能减少阶段间被自动选卡程序抢走的机会，不能阻止其他用户直接指定同一卡；没有独占调度器保证。

`--status` / 面板 `s` 显示保留 UUID，`g` 每 1 秒刷新。GPU 利用率降为 0% 不代表释放；GPU 阶段的数据加载期间仍保留；后续 CPU 尾部开始前已释放。正常完成、步骤失败或确认 `x` 中断时释放；流程主进程死亡后，保留进程自行退出。保留进程继承 runtime 锁并登记身份，更新与恢复会等待它们退出。记录位于本次 workflow 的 `gpu-holds/`，不写到代码目录。不支持在正在运行的旧进程中热切换此行为，更新后启动或续跑才生效。

选择“最多 4 张”表示并发上限。当前没有可用卡时，后台显示 `waiting_gpu（等待显卡）`，每 5 秒查询一次、约每 30 秒打印等待日志；有 1 张即可开始，2 张就并行两个独立阶段。后续候选卡空闲时自动加入剩余阶段，不会搬迁正在运行的模型。即使始终只有 1 张可用，也能按依赖顺序完成六阶段；若始终没有满足条件的卡，则会一直等待，可用 `--stop` 停止。

```bash
# 示例仅在已获准使用这些卡时执行；最多用 4 张，有 1 张就开始
bash start_server.sh --tasks train --gpu-count 4 --gpu 0,1,2,3 --non-interactive
bash start_server.sh --status
```

`--gpu auto` 在继承的可见范围中选择；显式列表进一步缩小候选范围。候选池可以比并发上限少，也可以更多。输入不存在、重复或越界的卡仍会报错，驱动查询和实际环境检查失败不会被伪装成资源等待。首次启动检查当前空闲卡，训练各阶段在其实际分配的卡上再执行环境检查，因此后来加入的卡也需要通过检查。

`--status` 展示等待状态或“正在使用 1/4 张卡、待运行 N 个阶段”；阶段日志和 `queue.json` 记录实际 GPU UUID。等待可被正常停止，之后仍用 `--resume-job` 恢复。实际训练失败（例如显存溢出）仍停止流程并保留检查点，不进行无限重试。

## 输出在哪里

每次提交生成 `<results>/<时间戳>_<标识>_workflow/`：

```text
plan.json                   # 所选步骤、有效参数与输入/输出路径
receipt.json                # 后台进程 PID 及启动身份
job_state.json              # 总体和逐步状态、退出码
launcher.log                # 完整后台日志
steps/
  01_preprocess/            # 按本次所选步骤编号
    status.json
    server-settings.json    # 只记录此步骤有关设置
    server.log
  02_train/
    environment/            # 所用设备的实际计算检查
    preflight/              # 正式尺寸预检
    train/                  # 根 models.json、folds.json、training-summary.json
      fold_0/               # 本折 models.json、六阶段权重、history
        stage_logs/         # 多卡时本折各阶段日志
        queue.json          # 本折多卡依赖和状态
      fold_1/
      fold_2/
  03_test/
    test/predictions/       # NIfTI 与预测记录
    test/evaluation/evaluation.json
    test/test-report.json
```

实际目录取决于所选步骤，示例中的编号不是固定菜单编号。缓存仍写入 `prepared` 指定位置。`<runtime>/last-job.json` 指向最近任务；`--status` 读取这一指针，旧任务的记录和日志仍保留。CPU 步骤不会新增 GPU 或无关模型设置，也不会改掉已保存的训练资源偏好；需要清除旧偏好时加 `--reset-settings`。

只有三折全部完成，根模型集合才标记 complete；单折配置仍要求六阶段完成。使用 `--resume-job` 恢复原流程，或菜单选 `r`：预处理/预测按病例接续；训练核验并保留已完成折，中断折从最后完整 epoch 接续；评估与诊断等短步骤保留旧目录后重跑。日志追加，每次恢复前的计划、状态和进程回执保存于 `attempts/`。普通新建任务仍创建新目录。详见 [恢复策略与命令](recovery.md)。

## 环境来源与验证范围

参考工作笔记 `zmic44_服务器基础信息_更新版9.22.md`，实际更新时间为 **2026-09-26**：Ubuntu 24.04.3、8×RTX3090 24 GiB、驱动 580.173.02、251 GiB 内存。硬件信息是历史记录，脚本会重新查询显卡。

复用已有 Conda 管理工具 `/data5/zhougaowei/zhangruichen_workspace/project1_runtime/tools/miniforge3/bin/conda`；CHD 使用自己 runtime 下的 `envs/chd_py311`，不向 whole heart 的 Python 环境安装依赖。默认 Python3.11、PyTorch2.8/CUDA12.8；未找到已有 Conda 时下载并校验 Miniforge。可用 `--python /已有环境/bin/python` 跳过安装，仍检查环境。网络设置沿用服务器环境。

已有验证覆盖 CPU 多进程训练、独立测试入口与模拟显卡派发。2026-10-10 增加真实 Linux 进程测试：父进程退出后继续运行、继承锁阻止重复提交、失败停止后续步骤、信号中断记录和前台退出码。进程测试采用合成任务；未连接 zmic44，尚未做真实多 GPU 或 3090 显存验证。当前论文规模配置 `configs/chd.yaml` 仍须在服务器做 preflight；多卡队列不拆分单阶段 batch4，真实单卡显存需求需实测。主干与默认三维位置关联见 [U-Net 说明](unet-source.md)。

## 独立诊断步骤

`diagnosis_server.sh --task diagnosis-labels/diagnosis-features/diagnosis-train/diagnose/diagnosis-evaluate/diagnosis-demo` 分别启动，均使用 CPU，不询问 GPU 或分割模型配置。也可将需要的步骤放入一次 `--tasks` 多选提交；标签表默认在 dataset 下的 `imageCHD_dataset_info.xlsx`，空白按约定默认转为 0。

训练诊断树需要 train/val 特征，因此首次诊断链路应使用全病例分割预测，不能只接 test 的输出。全病例预测和后续诊断一并提交的示例见 [独立诊断说明](diagnosis.md)。

## 非训练 CPU 分配与测试/预测多 GPU（0.13.0）

训练的原 CPU 调优、独立 smoke/preflight 步骤和训练代码保持原样。其余步骤启动时内置约 0.2 秒 CPU 采样，读取实际 affinity/cgroup、可用内存，按任务类型和并发上限选择线程预算。预处理、预测后处理、分割评估和诊断特征提取按病例并行；CPU-only 预测也可病例并行。最多 8 个 CPU 病例进程，多进程时每进程 1 线程，单进程最多 4 线程，再按负载和大病例内存估计收缩。是有界启发式分配，不是实测最优，也不是整机独占额度。

记录在步骤目录的 `cpu-allocation.json`、`.cpu-runtime/` 和 `progress.json`。不新增面板步骤，不重复训练的 180 秒调优。短小任务可能仅使用一个进程；病例进程启动有开销，真实吞吐仍需服务器计时。

测试和预测的 `--gpu-count` 也表示 1—5 张的上限，`--gpu` 仍是获准候选池。一卡一病例，保留 OOF/三折集成；单病例模型不会合并多卡显存。内存预算可能进一步收缩实际并发。少卡运行、无卡等待和继承 CUDA 限制不变。状态与病例日志见推理输出的 `queue.json`、`stage_logs/`。

面板步骤名称和编号不变。内部会先完成测试、预测及其它所选 GPU 工作，再做它们的 CPU 尾部和独立评估/诊断。前置环境检查、预处理和 GPU 推理必需的图像准备仍按依赖执行。`pending_cpu` 表示 GPU 计算已完成，等待 CPU 尾部；不等于整个测试完成。状态显示 GPU 释放时间。

GPU 中间结果是 `.gpu-results/` 下的未压缩工作网格，需额外磁盘空间；原网格恢复与压缩延后。`gpu-phase-ready.json` 绑定输入契约与文件哈希。CPU 尾部中断后，续跑先校验中间结果，有效则不重新申请 GPU；输入改变拒绝混用，损坏中间结果重新进入计算阶段。旧版已完成预测病例继续校验后复用。

## 大 project 中的独立快速验收

案例位于与 CT 并列的 `server-resource-smoke/`，本地为 `D:/code_project/server-resource-smoke/`。使用 6 例合成 CT、缩小三折六阶段模型，调用实际服务器入口，检查 CPU 记录、推理队列上限、OOF、GPU 释放顺序和残留进程。命令见 [案例说明](../../server-resource-smoke/README.md)。它不改正式数据或缓存，不证明真实 CT 显存/内存容量或医学效果。

## 历史结果查看（0.14.0）

更新代码并重新打开面板，输入 `o` → 选择项目/任务编号。这里的项目指已经命名的 workflow，例如 `CHD_1`；同名记录以目录区分。默认选中最近任务，也能选以前已完成、失败或仍运行的任务。`d` 可直接输入历史 workflow 的完整目录，`0/b` 返回，`q` 退出面板。

结果页读取既有 JSON 清单，不安装环境、不选择显卡、不改原文件或最近任务指针，不重新训练/预测/评估。旧版只有逐病例 Dice 时，按已保存的数值汇总各结构均值，忽略不适用的空值；没有评估结果时明确提示，而不是把完成训练显示为评估通过。

- 显示任务名、记录状态、完成步骤及结果目录。
- 分割显示平均前景 Dice、七个结构中文名与 Dice；已有三折记录时显示各折指标和推理方式。Dice 是重叠度，不能解释为诊断准确率；空值显示未记录/不适用。
- 训练显示单次划分或三折概况，预测显示清单记录数量和保存位置；短测/预检明确标注仅用于软件验证。诊断评估显示准确率和覆盖率，说明患者-疾病对及弃权分母。
- 缺少文件、损坏清单、仅完成 GPU 计算时都有提示；完整性哈希不在查询中重新计算。归档备份目录不当成当前结果。
- 页末直接给出结果摘要、训练明细和详细日志的可复制命令，路径中的空格和引号按 Bash 转义。

也可直接查询（使用系统 `python3` 即可）：

```bash
python3 start_server.py --query results --job-dir /原任务_workflow
# 只看测试结果
python3 start_server.py --query results --job-dir /原任务_workflow --step test
# 观察仍在运行的任务；Ctrl+C 只退出查看
python3 start_server.py --query results --job-dir /原任务_workflow --follow
```

结果选择器可从配置的结果根目录发现旧 runtime 的 workflow，查询不放宽原来的续跑资格。若列表没有旧任务，使用 `d` 或 `--job-dir` 指向它的 workflow 目录；无需移动原始影像、模型或结果。
