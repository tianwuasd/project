# CPU、内存与显卡调度

## 显卡选择

默认 `resources.share: false`，`min_gpus: null`。`--gpus 2` 等待两张满足显存安全预留的空闲卡。默认持续等待，`--wait-seconds 300` 可设置有限等待；Ctrl+C 取消。`--no-share-gpu` 可以覆盖已有配置中的共享设置。

`--gpu-ids 1,3` 限定物理编号及顺序，继承的 `CUDA_VISIBLE_DEVICES` 继续限制可用范围。`--share-gpu` 显式启用共享，并继续要求同配方、相同进程数量和每卡 batch 的实际显存峰值、连续低负载窗口和足够余量。共享记录不能通过修改数字代替实测。

需要自动调整卡数时设置 `--gpus 2 --min-gpus 1`。上限为两张，最少一张，优先达到目标数量。调整卡数需要对应 GPU、相同进程数量及每卡 batch 的同配方实测容量；条件不足继续等待。已有同规格完整轮次的运行可以提供记录。全局 batch、patch、间距、训练轮数和数据划分保持固定。手动 `batch_layout` 要求固定进程数量。

显卡重组发生在完整 epoch 检查点处。保存模型、SGD 动量、scaler、训练及验证进度、全局采样游标和逐 rank RNG；释放原组后重新准入。保留已有 rank 的随机状态，新增 rank 的种子由检查点状态、rank、进程数量和已消费步数确定。数据抽样由全局 slot 决定，未消费预取重放。损失继续聚合全局统计，再由 DDP 归约梯度。卡数变化可能改变浮点加和顺序，日志明确记录 `bitwise_equivalent: false`。同进程数量恢复仍校验完整状态和 RNG。

位于 epoch 内部的检查点需要原进程数量，等待相应空闲卡。自动增加卡数需要稳定观察间隔，默认 60 秒，在轮次边界复查。新增进程期间保留用户最少卡数和共享开关。

## CPU 与内存

`training/budget.py` 核算主机及当前 cgroup 剩余内存、项目内其他任务尚未兑现的预留、进程基础预算、同时构建病例的工作空间、每个实际 Dataset 实例的缓存、预取及传输副本。已有任务只用出生身份匹配的存活进程 USS 抵扣预留，避免重复扣除实际内存。

离线预处理的病例工作空间使用真实 NIfTI 头、清单确认的间距及冻结计划推导。初次估算采用裁剪前上限，日志中的 `cropped_shape: null` 表示尚未实际处理。病例运行日志另外记录实际裁剪和重采样尺寸。训练阶段使用离线数组形状、取块副本及前景位置字节核算，单独记录 mapped_file_bytes，不为整例重采样预留内存。估算值与进程 RSS 读数分别标明，预算不是硬内存限额。

所有训练 rank 共用 `BuildGate`，按当前取块和位置数组的估算领取内存额度，同时限制加载数。缓存按实际供数进程数计算，包括前景位置数组；零 worker 时使用 rank 内本地 Dataset，也纳入预算。最低配置仍不能容纳时等待，日志列出 `minimum_bytes`、`budget_bytes`、主机及 cgroup 来源。

离线预处理使用同一 `work_root/resources` 全项目登记 CPU/RAM 预留，不申请 GPU；病例完成后释放活动工作空间。worker 上限由 CPU、RAM、readers 和待处理病例数决定，与训练 batch 独立。无活动病例时按资源观察间隔重新规划，资源允许可增减 worker。实际派发再次检查实时可用量，仅抵扣身份核实的本组 USS。不同折并行启动时核算其他任务尚未兑现的预留，避免各自领取相同余量。

worker 数量受 `workers`/`max_workers`、当前全局 batch、各 rank 份额、CPU 比例及内存共同约束；worker 内部数值库单线程。两个训练 rank 可以在少量 CPU 配额上共享 CPU 时间，不要求额外一颗专用协调核心。配置上限为 32 个供数进程、8 个病例构建名额；全局 batch 4 时实际最多 4 个供数进程，每个 rank 最多使用自身份额数量的 worker。实际准入数量可以减少到零 worker、一例构建。

每个 rank 仅保留一个尚未消费的批次，应用后的 `prefetch` 固定为 1。当前全局批次需要的所有样本完成计算并释放输入后，才派发下一批次；病例和取块随机数仍由原全局 slot 协议决定。两卡、全局 batch 4 时，每个 rank 为当前批次准备两块数据。当前批次最多涉及四个病例，也可能重复抽到同一病例。

`HostCoordinator` 使用主机进程间条件变量协调进程启动、模型加载、CPU 数据就绪、GPU 传输完成、批次消费、供数池调整及检查点写入。所有 rank 完成启动后再建立通信组。数据准备期间，各 rank 在主机上等待，尚未进入训练 collective。只有全部 rank 的当前数据准备完成且传输完成，才进入模型计算。CUDA 异步复制通过事件确认完成后释放 pinned 输入；计算结束同步设备，再释放当前批次。NCCL/Gloo 通信超时由 `resources.collective_timeout_seconds` 设置，默认 180 秒，不用这个超时限制 CPU 预处理。

运行中默认每 30 秒重新规划，增加供数前至少间隔 60 秒。调整时停止派发，结束旧供数池，清理未消费预取和缓存，再检查实时余量并启动新池。GPU 模型、优化器和 scaler 保持驻留，全局采样游标保持不变。`supply_resize_proposed`、`supply_resize_applied` 分别记录提出及实际应用的配置。

父协调进程每秒检查有效主机内存；触发安全预留时请求完整保存并结束当前组。释放本组后缩小供数上限、缓存、预取和病例并发，保留缩小值重新准入，默认最多恢复三次。未达到最低预算时继续等待。未知错误、数据或身份错误直接终止并保留日志；外部瞬时 OOM 不保证能够提前拦截。

Linux 使用 systemd 获取当前进程的 cgroup，并通过 util-linux 的 findmnt JSON 解析真实挂载根；核验当前 PID 后逐层读取 cgroup v2 限额。Ubuntu 服务器需要 `libsystemd.so.0` 和 `findmnt`，无法核实当前 cgroup 时停止扩容。Windows 使用 psutil 的主机容量。

## 日志与验收

事件保存在运行目录的 `events.jsonl`，单份 16 MiB，加三份轮转；详细病例信息不会反复打印到终端。记录 requested/applied、主机和有效 cgroup 容量、逐病例工作空间、并发乘数、缓存实例、队列、实际 USS 抵扣及等待原因。`batch_enqueued`、`case_build_started/completed`、`cpu_batch_dequeued`、`batch_transfer_completed`、`batch_consumed` 包含批次序号、rank、病例或全局 slot；病例事件另含额度、缓存及耗时。`host_rendezvous` 心跳列出当前等待原因和已经就绪的 rank。训练记录供数、计算、验证、检查点耗时；短测另外记录进程组 RSS 峰值。未实际执行的 CUDA/NCCL 和 Linux cgroup 分支单独标明。

`tests/verify_resource_adaptation.py` 限定一个真实 ImageCHD 病例的 32³ 区域、全局 batch 4、总计两次更新，覆盖供数 1→0→1、确定性重放、内存不足准入、同进程恢复、完整轮次边界 1→2 个 CPU/Gloo 进程恢复及预测复用。临时文件由独占工作区自动清理，保留报告独立于工作区。

`tests/verify_ordered_supply.py` 限定两个真实 ImageCHD 病例的 32³ 区域、两个 CPU/Gloo 进程、全局 batch 4，总计两次训练更新及对应验证，检查恢复、每个全局 slot 恰好消费一次、单批队列上限和真实供数等待超过通信超时时仍能继续。显式 `--cuda-transfer` 使用一张通过空闲准入的显卡检查小块复制完成和缓冲释放，不进行 GPU 训练。

`tests/verify_offline.py` 在两例真实病例上检查离线处理、只读 mmap、完整复用、仅补缺失病例、稳定数据身份、两次更新/恢复、预测/评价和折统计隔离。`--preprocess-only` 只验证离线及共享资源登记，不执行训练。仿真选卡明确使用虚拟 GPU；真实计算使用 CPU/Gloo，不代表真实多卡 CUDA/NCCL 已验收。
