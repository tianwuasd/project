# STU-Net-S 多GPU执行（0.5.4）

采用 server-engineering skill V2 0.7.2 的固定工作量、全组资源准入、epoch边界成员调整和恢复原则。默认模型、1000轮、250/50次迭代、全局B4、128³、lr .001均保留。main与训练分支均包含训练。

## PyTorch 2.15设备绑定修复

2.15默认NCCL实现不再用`torch.cuda.set_device()`推断通信设备；未传`device_id`时按LOCAL_RANK或rank取模选择。资源调度可能给rank1分配可见cuda:5，因此仅设置当前设备和DDP.device_ids不足以保证通信组一致。0.5.2把实际准入的torch.device显式传给NCCL初始化，不把rank当GPU编号，也不改变继承的CUDA_VISIBLE_DEVICES或UUID准入。依据：[PyTorch 2.15官方变更说明](https://github.com/pytorch/pytorch/releases/tag/v2.15.0#distributed-c10d)。

rank启动会记录准入、设备/UUID、通信组初始化与DDP阶段；主launcher直接报告本代rank的fatal原因。尚无检查点时会明确显示，不声称已保留模型。此修复不改预处理、模型、loss或batch；已完成预处理仍按原合同校验复用。代码签名变化后使用新训练输出目录，保留失败任务的冻结代码，不能覆盖其code目录或强行resume。

本地CPU映射回归与单CUDA短测不能替代目标Linux、多GPU/NCCL的现场验收。无需仅因本问题重装环境；先修绑定并运行正式batch/patch短测。

## 启动

0.5.4 启动时，调度进程完整核验输入。每次创建训练组，各路分担影像和标签的 SHA256 及前景坐标检查，每个病例都有一路负责。各路分别校验自己读取的属性文件并核对共同的数据身份。任一路读到损坏或不同的输入，全组都会收到错误。

查看各路 `training.step` 事件中的 `wait_seconds` 可以判断哪一路在等下一批数据。`supply.stages_seconds` 分别记录等待读取许可（`read_gate`）、读取解压及裁剪（`read_decode_crop`）和 CPU 增强（`augmentation`）。这些时间属于并行供数的工作进程。比较配置时用 `epoch-performance.jsonl` 中 `complete_from_boundary=true` 的完整轮次。

0.5.3检测到本组GPU出现外部计算PID后，在当前同步步完成处保存完整状态、释放本组，再选择获准空闲卡：优先保留可用成员，补充其他空闲卡；没有足够备用卡就减卡，仍保持同一个全局batch。无卡时在配置的有界等待后保存`stopped / reason=no_available_gpu`。`--no-elastic`仅关闭主动扩卡，不关闭换卡/减卡；换卡不会越过本次已采用的卡数上限。用户SIGINT/TERM仍明确停止，不自动重启。

半轮train或validation都可从已提交的样本序号继续。相同成员精确恢复逐rank RNG；成员变化时按保存的游标和新成员确定性设置rank RNG，实际样本和增强仍由原全局槽种子决定。不同卡数的浮点归约不承诺逐位一致。CPU/Gloo4→2→1→3数值/优化器/scaler/游标验证不能替代真实Linux多CUDA抢占验收。

0.5.2 或 0.5.3 的训练要启用新代码，先让旧任务安全停止并确认检查点已提交，再从更新后的项目目录运行 `run.py train`，沿用原数据、折和训练参数，添加 `--continue-from /旧训练run --output /新训练run`。使用全新的输出目录，旧运行保留原有的冻结代码。程序验证两个正式发布源在 CPython 3.11 和 3.12 上的对应指纹、数据、配方、环境和作者权重身份，再核验并复制 latest、epoch 及已有的 best 完整状态。新目录中的 `continuation.json` 记录实际接续的轮次和步数。新运行再次暂停时使用它自己的 `code/run.py` 和 `--resume`。

训练记录每一路等待下一批数据的时长。在完整轮次结束、预取队列排空后，调度器根据连续采样窗口和最新 CPU、内存余量尝试调整供数进程。训练日志中的 `supply.feedback` 记录调整前后的分配、等待时间和原因。显式传入的 `--workers` 作为整组进程数上限，含义与内存保护触发后的临时缩减额度分开。短于 40 个训练步的单轮可测试基本训练流程，供数反馈验收请至少覆盖 40 步和多个完整轮次。

旧任务若经历过内存保护，随后又省略 `--workers` 续跑，导入器会读取旧监督日志。它区分原始准入、内存保护减半，以及后来可观察到的人工降额。日志完整且能还原自动配置时，新任务可在资源恢复后试增供数。证据缺失时沿用旧额度作为上限。`continuation.json` 中的 `source_worker_limit` 和 `source_worker_limit_evidence` 记录判定结果。若需明确指定上限，迁移命令可带 `--workers N`。

资源在准入期间被抢占会重新选卡；运行中若抢占导致通信/OOM失败，仅在有明确资源证据时从最近完整提交恢复，未提交的工作可能重算。数据、合同、未知错误和无法承载固定batch/patch的容量问题仍留证报错，不以无限重试掩盖。所有进程清理只处理核实属于本次任务的进程，不终止外部PID。

```bash
# 复用已通过training检查的现有环境；独立training环境则改为实际路径
py=.venv_preprocess/bin/python

# 检查全部输入与配方，不初始化CUDA或申请显存
"$py" run.py train --prepared /预处理run --gpus auto --check-only

# 原尺寸缓存取64³patch的短测。默认B1，只能使用1张卡
"$py" smoke.py --stage train --prepared /预处理run --gpus 1 --output /结果/short

# 显式多卡短测：全局B4仍为短测64³，不代表正式128³通过
"$py" smoke.py --stage train --prepared /预处理run --batch-size 4 --gpus auto --output /结果/multi-short

# 正式配方：仅使用授权可见卡，不自动改大batch
CUDA_VISIBLE_DEVICES=0,1,2,3 "$py" -u run.py train \
  --prepared /预处理run --fold 0 --gpus auto --output /结果/fold0

# 相同目录继续；恢复时不要再带stop-after-steps
"$py" -u run.py train --prepared /预处理run --gpus auto --output /结果/fold0 --resume
```

`--gpus N`为最多N张卡，`auto`最多为全局batch；有可靠UUID、空闲PID及显存证据才准入。B4布局为[4]/[2,2]/[2,1,1]/[1,1,1,1]，不会使用第五张非空rank。多卡要求Linux/NCCL；本机Windows单CUDA短测没有多卡DDP包装。`--device`用于CPU或legacy单设备；新多卡路径用CUDA_VISIBLE_DEVICES限定范围，不猜物理编号。

`--workers N`是所有rank合计的供数进程上限，0为各rank串行取块。统一CPU/RAM80%准入，先扣监督器、GPU rank、串行/驻留输入预算，再分配持久worker。每个worker内层线程1、至少一个派发槽，通常2倍预取；文件读取使用全组独立闸门。显存预留max(4GiB,总显存10%)，每rank实际local batch前反向及动量容量通过后才开始训练。容量不足明确停止，原batch/patch保留。

## 正确性和恢复

缺MYO病例的源0仍监督P0+P5；CE全组按有效体素加权，Dice先按样本屏蔽缺标类别，再归约每类充分统计。采用可微归约，抵消DDP参数梯度平均的world因子。不能简单平均各卡loss。全ignore的rank仍参与所有collective，全组无监督loss为可微0。FP16溢出在全rank同步后跳过更新，消费一次尝试，不补取样本。

每个patch的种子由原批次序号及全局槽号确定，换卡数只分配槽，不重复生成整个B4；前景槽仍按全局B判断。它是新执行协议，和旧单卡串行RNG序列不同，已记录为科学合同的一部分。连续/恢复可按同成员验证精确重放；成员数改变不承诺浮点逐位一致。

检查点保留权重、SGD动量、scaler、阶段/步数、已消费游标、成员UUID及逐rank RNG。0.5.3允许同步步完成后的成员更换；未消费预取重放。旧训练继续使用`旧run/code/run.py`及原环境/参数，或通过上述受校验的显式继续训练入口切换策略；不热修旧冻结目录。

默认每20秒只读核查可见设备的PID/显存，不另建常驻CUDA keeper；空闲候选经两个稳定观测后请求加入，在下一完整epoch提交后才退出原组并重新准入。`--no-elastic`关闭扩卡，但保留外部PID检查。原组至少4轮、末3轮墙钟中位数为基线；扩卡首轮暖机，其后连续3轮均慢25%以上才回原组，新增卡本阶段延期。样本不足不宣称退化。

主机/cgroup内存每秒采样。持续低于阶段入口余量（20%、至少4GiB）时整组退出，最多三次缩减全组worker上限，从最近完整epoch重算未完成部分；减少的预算和入口预留跨重组保留。外部瞬时OOM不保证可被采样拦住。进程清理核对本次Popen/worker登记的PID出生身份，不清理其他项目或全局SHM。

Ctrl+C/TERM向监督器提出安全暂停；Linux可用nohup/tmux运行，普通前台命令不保证断开SSH后继续。任何rank失败会停止同组，保留已提交检查点。没有完成容量检查前失败时，目录可能只有签名/快照，无模型检查点，重新启动仍需作者权重。

## 日志与短测

- `status.json`、`logs/run.log`及`logs/events.jsonl`：rank0的全局进度、loss、实际更新/溢出。
- `ranks/rank_N/logs/`：各rank供数、等待、实际worker PID、主机内存；`capacity_probe.json`：实际local shape与显存峰值。
- `control/logs/`：监督器等待、requested/applied、布局、UUID、重组原因；`control/job-*.json`和`exit-*.json`保留每次启动资源与退出证据。
- `epoch-performance.jsonl`：含验证/检查点/同步的完整epoch墙钟。中途恢复的半轮明确标记不可比较。
- `checkpoints/latest.json`原子指向完整状态；`epoch.json`单独保留最近的完整轮次，避免多次中途暂停挤掉压力恢复所需状态。best/final指针与`history.json`、`summary.json`保留。

```bash
# 普通回归
python -m unittest discover -s tests -q
# 真正CPU/Gloo DDP：1/2/3/4rank的值、梯度、SGD更新、同步溢出与RNG恢复
python tests/distributed_smoke.py --output /独立验收目录/gloo
```

CPU/Gloo与本机单GPU短测不能替代目标Linux CUDA/NCCL、正式128³/B4容量或持续提速验收。本版未开启torch.compile，也没有已测得的加速倍数。三折仍分别训练；本次优化的是一折内的固定batch分担，没有把三折混为同一模型。
