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
| `gpu_count` 训练 GPU 数 | `1`，可设 1—5 |
| `gpu` 指定卡 | `auto`；也可写 `0,2`，数量须与 gpu_count 一致 |

原始数据默认路径是本项目约定位置，尚未确认已上传；数据必须已解压，符合 ImageCHD 的 image/label 配对规则。不能把 Wholeheart_Train_Dataset 直接当 ImageCHD 输入。路径不正确时用 `--dataset /实际路径/ImageCHD_dataset` 覆盖。

设置优先级：**命令行 > runtime 中记住的设置 > 服务器配置文件**。保存在 `<runtime>/.server-settings.json`，每一步另存只含该步骤有效参数的 `server-settings.json` 快照。更换配置默认值后，用 `--reset-settings` 忽略上次选择；不删除缓存或模型。模型结构和 gate 开关仍在 CHD YAML 中。

## 菜单多选，默认后台运行

在服务器的 `project/chd-ct-reproduction` 中执行：

```bash
bash start_server.sh
```

输入 `0,1,2,4,5`，即环境检查 → 预处理 → 软件短测 → 正式训练 → 测试；逗号、空格或中文逗号均可。已有预处理缓存时选 `0,2,4,5`。编号是服务器菜单的编号，桌面 `start.py` 仍为单选。

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
  --gpu-count 2 --gpu 0,1 --non-interactive
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

训练只读缓存，测试只读已有模型；无标签 CT 使用独立预测，不能做需要真值的测试/评估。预处理目录已存在会停止，需取消预处理步骤以复用缓存。正式训练失败不会将不完整模型记为上次成功模型。

## 多卡怎样工作

沿用 whole heart 的**一张卡一个模型任务**方式，不使用 DDP，也不把多张显卡显存合并。

- `crop64`、`crop128`、`all64`、`all128`、`blood2d` 可独立训练，最多五个同时运行。
- `blood_lstm` 必须等待本次 `blood2d` 成功，使用并冻结其编码器。
- 子进程只可见自己分配的 GPU；开始下一阶段前重新检查该卡。任何阶段失败会停止新派发并结束本队列仍在运行的子进程，保留日志与已经完成的权重。
- 同时运行的数量受 GPU 数和可运行阶段数共同限制，后期可能只有一张卡在工作。
- 预处理、缓存检查和评估用 CPU。融合预测/测试用一张卡；若保存了多个显式编号，使用列表第一张。`gpu_count` 控制训练并发数及环境检查卡数。
- 每个训练阶段采用 `seed + 阶段序号`，使单卡顺序运行与多进程运行有一致的随机种子。它改变旧版本后续阶段的随机数轨迹，但不改变网络或已有权重的加载格式。

保持尊重 `CUDA_VISIBLE_DEVICES`；空闲卡不足、卡上已有计算进程、低于 6000 MiB 空闲显存或利用率高于 10% 时停止，不自动降卡数或切到 CPU。空闲检查不是集群调度器预约，共享服务器仍应先确认分配。

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

只有六阶段全部完成，模型集合才标记 complete。中途失败不支持自动断点续训；排查日志后重新提交会创建新目录，不覆盖已有输出。

## 环境来源与验证范围

参考工作笔记 `zmic44_服务器基础信息_更新版9.22.md`，实际更新时间为 **2026-09-26**：Ubuntu 24.04.3、8×RTX3090 24 GiB、驱动 580.173.02、251 GiB 内存。硬件信息是历史记录，脚本会重新查询显卡。

复用已有 Conda 管理工具 `/data5/zhougaowei/zhangruichen_workspace/project1_runtime/tools/miniforge3/bin/conda`；CHD 使用自己 runtime 下的 `envs/chd_py311`，不向 whole heart 的 Python 环境安装依赖。默认 Python3.11、PyTorch2.8/CUDA12.8；未找到已有 Conda 时下载并校验 Miniforge。可用 `--python /已有环境/bin/python` 跳过安装，仍检查环境。网络设置沿用服务器环境。

已有验证覆盖 CPU 多进程训练、独立测试入口与模拟显卡派发。2026-10-10 增加真实 Linux 进程测试：父进程退出后继续运行、继承锁阻止重复提交、失败停止后续步骤、信号中断记录和前台退出码。进程测试采用合成任务；未连接 zmic44，尚未做真实多 GPU 或 3090 显存验证。原作者宽度配置 `configs/chd-author-unet.yaml` 仍须在服务器做 preflight；主干来源与三维默认 gate 见 [U-Net 说明](unet-source.md)。

## 独立诊断步骤

`diagnosis_server.sh --task diagnosis-labels/diagnosis-features/diagnosis-train/diagnose/diagnosis-evaluate/diagnosis-demo` 分别启动，均使用 CPU，不询问 GPU 或分割模型配置。也可将需要的步骤放入一次 `--tasks` 多选提交；标签表默认在 dataset 下的 `imageCHD_dataset_info.xlsx`，空白按约定默认转为 0。

训练诊断树需要 train/val 特征，因此首次诊断链路应使用全病例分割预测，不能只接 test 的输出。全病例预测和后续诊断一并提交的示例见 [独立诊断说明](diagnosis.md)。
