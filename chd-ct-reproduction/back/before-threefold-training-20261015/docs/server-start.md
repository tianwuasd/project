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
| `s` / `g` / `p` | 流程状态 / 显卡状态（含进程占用）/ 本用户的工程进程 |
| `m` | 六个模型的已完成 epoch、目标轮数、train/val loss、学习率及 GPU UUID |
| `l` | 选择步骤和模型查看详细日志；可看一次或持续刷新 |
| `x` | 停止正在运行的任务；先显示目标名称和目录，再确认 |
| `h` | 选择历史任务，查询其状态和日志 |
| `n` | 无任务运行时新建，输入本次任务名称 |
| `r` | 无任务运行时，从历史列表选择中断/失败任务续跑 |
| `0` | 退出面板，后台任务继续 |

新建后输入 `0,1,2,4,5`，即环境检查 → 预处理 → 软件短测 → 正式训练 → 测试；逗号、空格或中文逗号均可。已有预处理缓存时选 `0,2,4,5`。下表是**新建任务后的步骤菜单**，与上面的任务管理面板分开，桌面 `start.py` 仍为单选。

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

程序按表中顺序执行所选步骤，去重，不自动添加未选步骤。提交前显示输入、输出和执行顺序；缺少必要输入会直接停止。同一次流程中，后续步骤自动接用前面生成的结果，例如测试使用本次正式训练的模型。显式传入的 `--models`、`--features` 等优先。正式训练自身始终先做真实尺寸预检。

非交互运行同一流程（例子假定已获准使用 GPU 0、1）：

```bash
bash start_server.sh --tasks environment,preprocess,smoke,train,test \
  --name '首次训练' --gpu-count 2 --gpu 0,1 --non-interactive
```

已有缓存时去掉 `preprocess`；更换数据来源加 `--dataset /实际路径/ImageCHD_dataset`。需要保留原缓存时给新数据指定新的 `--prepared`。仅预处理不需要显卡参数：

```bash
bash preprocess_server.sh --non-interactive
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

## 随时查询模型进度和详细日志

在工程目录里直接运行。查询只需系统 Python，不安装训练环境、不占训练锁、不触发选卡或新任务；使用自定义 runtime 时，追加同一个 `--runtime /完整路径`。

```bash
# 当前流程、显卡和模型训练记录
bash start_server.sh --query status
bash start_server.sh --query gpu
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

日志中的 loss 在每轮训练及验证都完成后才更新，没有逐 batch 百分比或可靠剩余时间。队列心跳只表示调度器还在运行；`blood_lstm` 显示依赖等待时需要先完成 `blood2d`。恢复前的旧队列会明确标注，不能据其判断当前进度。无模型独立日志时暂时显示步骤 `server.log`，持续查看会发现后来生成的模型日志。首次日志读取最多取每个文件末尾 256 KiB，避免大日志拖慢查询。

查询和面板中的 Ctrl+C 只退出查看/返回面板；`--foreground` 下 Ctrl+C 才会中断任务。停止任务请用面板 `x` 或 `--stop`，停止请求后再次查状态，确认进程退出再续跑。

## 每一步仍可单独运行

下面是独立入口示例，**按需选择一条提交，完成后再提交下一条**；要连续完成多步，请使用上面的 `--tasks`。

```bash
bash smoke_server.sh --gpu-count 2 --gpu 0,1 --non-interactive
bash train_server.sh --mode train --gpu-count 2 --gpu 0,1 --non-interactive
bash test_server.sh --models /完整路径/某次正式训练/train --non-interactive
bash predict_server.sh --models /完整路径/某次正式训练/train --split test --non-interactive
bash evaluate_server.sh --predictions /完整路径/某次预测/predict --split test --non-interactive
bash train_server.sh --mode check --non-interactive
```

`smoke_server.sh` 使用缩小网络，每阶段一个训练 batch 和一个验证 batch；`test_server.sh` 使用已有模型做测试集推理和 Dice 评估。只有短测权重时，明确加 `--allow-smoke`，结果仅验证软件。`train_server.sh` 未指定模式时默认短测；多选菜单中的“4 正式训练”及 `--tasks train` 默认正式训练，也可显式传 `--mode` 覆盖。

训练只读缓存，测试只读已有模型；无标签 CT 使用独立预测，不能做需要真值的测试/评估。预处理目录已存在默认停止；完整缓存可取消预处理步骤直接复用，中断缓存可加 `--resume` 校验接续。正式训练失败不会将不完整模型记为上次成功模型。

## 多卡怎样工作

沿用 whole heart 的**一张卡一个模型任务**方式，不使用 DDP，也不把多张显卡显存合并。

- `crop64`、`crop128`、`all64`、`all128`、`blood2d` 可独立训练，最多五个同时运行。
- `blood_lstm` 必须等待本次 `blood2d` 成功，使用并冻结其编码器。
- 子进程只可见自己分配的 GPU；开始下一阶段前重新检查该卡。任何阶段失败会停止新派发并结束本队列仍在运行的子进程，保留日志与已经完成的权重。
- 同时运行的数量受 GPU 数和可运行阶段数共同限制，后期可能只有一张卡在工作。
- 预处理、缓存检查和评估用 CPU。融合预测/测试从候选池中选择一张空闲卡。`gpu_count` 是训练并发上限，环境检查至多检查这么多张当前空闲卡，不要求凑齐。
- 每个训练阶段采用 `seed + 阶段序号`，使单卡顺序运行与多进程运行有一致的随机种子。它改变旧版本后续阶段的随机数轨迹，但不改变网络或已有权重的加载格式。

菜单选择显卡前先显示实时物理编号、型号、空闲显存、利用率、计算进程占用和可见范围。执行时重新检查，不将菜单快照当作预约。保持尊重 `CUDA_VISIBLE_DEVICES`；卡上已有计算进程、低于 6000 MiB 空闲显存或利用率高于 10% 的卡暂不派发。不会切到 CPU、放宽门槛或扩展分配范围。空闲检查不是集群调度器预约，共享服务器仍应先确认分配。

### 显卡不足时等待和少卡运行（0.8.0）

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
    train/                  # models.json、权重、history
      stage_logs/           # 多卡时各阶段日志
      queue.json            # 多卡依赖和状态
  03_test/
    test/predictions/       # NIfTI 与预测记录
    test/evaluation/evaluation.json
    test/test-report.json
```

实际目录取决于所选步骤，示例中的编号不是固定菜单编号。缓存仍写入 `prepared` 指定位置。`<runtime>/last-job.json` 指向最近任务；`--status` 读取这一指针，旧任务的记录和日志仍保留。CPU 步骤不会新增 GPU 或无关模型设置，也不会改掉已保存的训练资源偏好；需要清除旧偏好时加 `--reset-settings`。

只有六阶段全部完成，模型集合才标记 complete。0.7.0 起使用 `--resume-job` 恢复原流程，或菜单选 `r`：预处理/预测按病例接续，训练从最后完整 epoch 接续；评估与诊断等短步骤保留旧目录后重跑。日志追加，每次恢复前的计划、状态和进程回执保存于 `attempts/`。普通新建任务仍创建新目录。详见 [恢复策略与命令](recovery.md)。

## 环境来源与验证范围

参考工作笔记 `zmic44_服务器基础信息_更新版9.22.md`，实际更新时间为 **2026-09-26**：Ubuntu 24.04.3、8×RTX3090 24 GiB、驱动 580.173.02、251 GiB 内存。硬件信息是历史记录，脚本会重新查询显卡。

复用已有 Conda 管理工具 `/data5/zhougaowei/zhangruichen_workspace/project1_runtime/tools/miniforge3/bin/conda`；CHD 使用自己 runtime 下的 `envs/chd_py311`，不向 whole heart 的 Python 环境安装依赖。默认 Python3.11、PyTorch2.8/CUDA12.8；未找到已有 Conda 时下载并校验 Miniforge。可用 `--python /已有环境/bin/python` 跳过安装，仍检查环境。网络设置沿用服务器环境。

已有验证覆盖 CPU 多进程训练、独立测试入口与模拟显卡派发。2026-10-10 增加真实 Linux 进程测试：父进程退出后继续运行、继承锁阻止重复提交、失败停止后续步骤、信号中断记录和前台退出码。进程测试采用合成任务；未连接 zmic44，尚未做真实多 GPU 或 3090 显存验证。当前论文规模配置 `configs/chd.yaml` 仍须在服务器做 preflight；多卡队列不拆分单阶段 batch4，真实单卡显存需求需实测。主干与默认三维位置关联见 [U-Net 说明](unet-source.md)。

## 独立诊断步骤

`diagnosis_server.sh --task diagnosis-labels/diagnosis-features/diagnosis-train/diagnose/diagnosis-evaluate/diagnosis-demo` 分别启动，均使用 CPU，不询问 GPU 或分割模型配置。也可将需要的步骤放入一次 `--tasks` 多选提交；标签表默认在 dataset 下的 `imageCHD_dataset_info.xlsx`，空白按约定默认转为 0。

训练诊断树需要 train/val 特征，因此首次诊断链路应使用全病例分割预测，不能只接 test 的输出。全病例预测和后续诊断一并提交的示例见 [独立诊断说明](diagnosis.md)。
