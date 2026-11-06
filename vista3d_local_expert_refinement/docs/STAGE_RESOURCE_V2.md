# 0.6.0数据协议补充

资源与恢复机制继续适用；数据范围以[FULL_SOURCE_TRAINING.md](FULL_SOURCE_TRAINING.md)为准：默认30训练/8验证/20独立测试，原生随机取块保持。本文历史fit8/prepare12记录不代表新运行数量，旧验证JSON保留原版本证据。

# 分阶段资源执行 v2（面板 0.5.0）

本版将预处理、协同训练、预测及 CPU 收尾接入各自的现场资源决策。发布代码已经做本地控制与真实 CPU API 验证；这不是目标服务器部署、真实 CT 效果或 CUDA/NCCL 性能验收。机器可读证据和运行文件摘要见 `STAGE_RESOURCE_V2_VALIDATION.json`。

## 不变的科学与采样合同

CT 仍为 seed0、250 epochs，每轮 **250 次训练尝试和 50 次验证迭代**。来源 fit/cal、标签、计划 patch/spacing/resampler、Dice+CE/深监督、SGD/PolyLR、正式 best checkpoint、原网格指标保持原合同。自动 batch 已获准；global batch 改变时，每轮训练/验证采样分别为 `250 × global batch` / `50 × global batch`，状态页同时显示原参考采样量。不会为了凑足 250 次成功参数更新而追加取批。

AMP 有限 loss 的梯度溢出属于正常缩放恢复：先在全部 rank 汇总非有限梯度标志，任一 rank 溢出则全部跳过 optimizer.step，统一降低 scale 并重置增长计数。该输入仍已消费、计入尝试和采样；日志明确记录跳过。非 AMP 非有限梯度、非有限 loss、OOM 和未知故障仍使整组失败；OOM 不允许局部 rank 跳过后继续。CPU GradScaler/Gloo 验证了单侧注入的溢出、统一回退、动量不变和下一次成功更新，不能代替真实 CUDA AMP 验证。

进度、失败记录和检查点分别记录 `attempted`、`successful`、`amp_skipped`。只有真正执行的参数更新计入 successful。短测原有 12 次尝试加恢复后 1 次尝试保持不变；其实际更新数不再硬编码为 13，恢复尝试被 AMP 跳过时不宣称恢复更新验证通过。容量推断仍要求实际存在的 optimizer state，不能用“尝试过”替代初始化证据。旧检查点没有计数字段时，从恢复点重新累计并保留历史未知标志；不凭 epoch 推算过去的成功更新数。

## 预算、供数和 GPU 生命周期

每个阶段使用当前获准且可用 CPU 的 `floor(0.80 × shares)`，以及当前可用 RAM 的 80%；cgroup 祖先配额、有效 affinity、已知零与未知容量分开处理。不能把全主机空闲核当成当前获准核。准备、训练、预测、CPU 收尾及训练整组重启重新观察；训练运行中的反馈保留阶段入口预算，避免把自身占用误判为外部压力。运行中新增外部容量与外部压力归因仍缺实测，不宣称持续 OS 资源预留。

训练一次直接给配置，只有有界容量恢复及两段有效供数等待窗口触发调整，没有候选速度排名，也不复制测评 UUID/batch/workers 到正式任务。额外资源没有收益时保留。主线程、交替执行的训练/验证线程和同时驻留的两个 worker 池、队列、主/固定缓冲均入账。RAM 先限制实际 worker 数后，若同一 worker 数能容纳已支持的 2 线程，重新采用 2；例如 CPU23、RAM23×64MiB、batch64MiB 为 W5/T2，总训练槽14。CPU191 的充足 8 rank 配置仍各 W16/T1；不为提高线程数牺牲 worker 数，反馈上限同步受约束。

持久 worker 跨 epoch 保持 PID；有界共享传输可回退到保真管道。预取但未消费输入在暂停、保存、失败和重配时按逻辑游标重放；AMP 正常跳过属于已消费输入。正式 epoch/phase 限制避免越界生成。固定官方 loader 的本地适配器将内层单线程限制替换为获准线程数；实际 load_case 与原 crop_and_pad_nd 的延迟切片读取经过读槽，增强在槽外。保留原裁剪、采样和增强实现、源码摘要及实际内层观察，不改 site-packages。CUDA 两槽 pinned transfer/事件和编译兼容性保护仍需目标 GPU 实测。

继承 GPU 可见性并排除外部 PID，所有已取得且兼容的空闲卡合作；训练新增卡只在完整 epoch 检查点边界加入。预测每卡保持模型，病例独立提交，最多一个下例准备槽；晚到空闲 GPU 可接剩余病例。所有拥有的 GPU 进程与 keeper 退出后才开始 CPU 收尾。收尾预算读取已提交 logits 的实际 NPY 头/文件长度与原图/输出几何，纳入模型网格，拒绝截断文件；不加载体素做入场估算。估算仍是保守先验，不是实测瞬态峰值。

## 升级后的来源与诊断

准备身份统一核对 `server_prepare.py`、`data_pipeline.py`、`prepare_nnunet.py`、`case_execution.py`、`read_admission.py`、`case_memory.py`、`cpu_case_queue.py` 七个摘要。新依赖同时进入生产者、来源检查与菜单共享校验。版本不兼容的来源在菜单标为不可用；更新后在 1 新建预处理，再在 2 选择新来源、3 选择训练来源。训练板块不会暗中重建预处理。旧任务保持冻结快照，不能以新 release 对旧缓存做任意签名迁移。实际本地 Git 双更新/隔离 release、新任务创建及 prepare→train→predict 来源图由控制测试覆盖。

日志选择对嵌套 worker 使用相对路径的唯一 key/label，平面日志旧 key 保持兼容；选中一个 worker 不会混入同名另一 worker。错误摘要保留关联 rank 根因及 wrapper 两条路径。菜单 z 的 ZIP 只包括任务拥有的文本诊断：资源/阶段决定、实际应用值、缺失证据、进度、采样、rank stdout/stderr、完整异常、反馈、病例观察、失败上下文及摘要清单。它排除数据、权重、缓存、环境和凭据，脱敏常见令牌，拒绝链接/重解析/越界路径；活动日志变更则拒绝并提示重试。ZIP 只写本地，位置为 `runtime/diagnostic-exports/diagnostics-时间-随机标识.zip`。

u 保持认证 GitHub 的代码对象按需获取、完整库存验证、禁隐式联网归档、隔离 release 和原 runtime 连续性。w/v 保持原预览、授权范围及上传边界。当前提交仅为本地项目交付；主会话负责保留并发上游、更广审查、发布及随后 Skill 同步。此文不表示已推送或部署。

## 已验证的效果与限制

实际 pinned nnU-Net2.8.1 loader CPU 记录证明串行/并行数据与目标一致、恢复游标一致、内层 Torch/native 两线程生效及跨 epoch PID 不变。合成供数记录 `task2-supply-effect.json` 中稳态 epoch1 平均取批等待由 0.071924s 降至 0.009756s，同时首批启动由 0.161524s 增至 4.688944s；两个 worker PID 保持，12 个批次全部共享传输、8 次缓冲复用、关闭后共享字节/PID 为零。只是该合成 CPU 供数负载，不能换算成端到端训练倍数。

真实 CPU 合成预处理已完成 fit-only fingerprint8、planning、正式12例、复用、预测准备和测评子集3；Blosc threads1/3/7 解压数组及压缩配置一致，原网格/仿射/空病例指标接口通过。Task2/Task5 原证据保留；本次最终修复改变了读取与导出边界，已重新执行受影响的真实 CPU 预处理、加载与检查点 API，见后文。Task6 对改变的训练/检查点路径做新的 AMP、Gloo、恢复和成员变更检查，并执行最终完整控制套件一次。

本机 Python3.11.15、Torch2.12.1+cu126、nnU-Net2.8.1，与目标预期 Torch2.8.0+cu128 不同。只读显卡查询仍有外部 PID，本次没有 GPU 计算。Windows 不提供真实 Linux cgroup/NUMA/共享内存/进程组证明；符号链接创建不可用的测试明确跳过，独立重解析标志/越界拒绝通过。部分 CUDA/依赖测试条件跳过，不能写成通过。

目标验收须在用户允许的空闲 GPU 上，使用真实尺寸/获准数据验证：CUDA AMP、分配/保留内存分离、编译及回退、pinned stream、NCCL 全组错误与成员变更、每阶段 CPU80/RAM80、实际内层线程/PID、病例保留与 GPU 释放后 CPU 收尾、端到端吞吐和科学输出。收集 z 诊断包；不提交原始数据或权重作为代码。

本地复核命令（项目目录内；环境与绝对证据路径见 JSON）：

Task6 修复前基线完整控制套件：**339 项，295 通过、44 条件跳过、0 失败，85.874s**，日志 `task6-panel-suite-release-green.log`。44 跳过分为：32 项缺可选 CPU Torch/Gloo/pinned API 依赖，6 项需 Linux `/proc`/Bash/flock 或 POSIX 描述符，4 项 Windows 无符号链接创建权限，2 项需影像/NIfTI API。实际 CPU API 的独立通过记录另列，不能把这些控制套件跳过项统称 GPU 跳过。此前未加入口保护的进程递归、旧预算夹具和时间戳并列夹具失败均保留日志；后两次具体失败修复后才重跑发布汇总，没有篡改原结果。

```powershell
$env:PYTHONPATH='tests;05_src'
$workRoot = 'D:/projects/back'
$run = 'vista-v2-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
$scratch = Join-Path $workRoot "t17/$run"
$reportDir = Join-Path $workRoot "sync-reports/$run"
$apiPython = Join-Path $workRoot 'sync-reports/vista-api-runtime-20261027/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $apiPython -PathType Leaf)) { throw "找不到复核环境 $apiPython" }
New-Item -ItemType Directory -Path $scratch, $reportDir -ErrorAction Stop | Out-Null
$env:TMP = $scratch
$env:TEMP = $scratch
$env:PYTHONUTF8='1'; $env:CUDA_VISIBLE_DEVICES='-1'
& 'D:/programtools/anaconda/python.exe' -B tests/run_panel_suite.py
& $apiPython -B -m unittest test_amp_updates test_execution_transfer -v
& $apiPython -B tests/amp_cpu_gloo_smoke.py --report-file (Join-Path $reportDir 'task6-amp-gloo-authorized.json')
& $apiPython -B tests/nnunet_distributed_checkpoint_api_smoke.py --report-file (Join-Path $reportDir 'task6-checkpoint-api.json')
& $apiPython -B tests/nnunet_membership_checkpoint_api_smoke.py --report-file (Join-Path $reportDir 'task6-membership-api.json')
```

Gloo、本地 Git 与子进程测试需要原已授权的本机执行权限；受限执行的空日志超时保留为环境失败证据，不通过弱化断言或放宽路径保护修复。所有本地 API 均仅 CPU 合成数据；上述命令不是服务器正式训练指令。


## 全项目最终审查修复（2026-10-28）

四项发现已在同一代码波次处理：

- 预测准备改为队列 supervisor 直接启动的 CPU 进程，独立 Popen 句柄、进程注册和 POSIX pass_fds 锁传递。GPU worker 不再生成无监管的孙进程。GPU 异常退出后分别清理 GPU 与准备进程；不向已退出 leader 的旧进程组盲发信号。请求/响应在系统 TMP，退出时归档小控制记录到持久队列日志；失败诊断指向真实持久 worker.log。
- read_admission 使用 TMP 下的 OS 文件锁槽，实际 payload 读取结束立即释放；嵌套 read_seg→read_images 可重入。正式/测评准备、官方 fit 指纹、预测准备、导出复制/哈希/标签读取都使用槽。计算 worker 与线程分配独立，已有重采样、压缩、统计和增强不会因为读限制而一律降到两个计算进程。默认 TMP 应位于本机快速盘；没有据此宣称所有服务器 TMP 的存储类型都已验证。小头信息读取不计入大 payload 限流。
- 训练窗口记录真实 host PSI（若可读），明确它不能诊断此 loader 的 storage/IPC 因果。两有效高等待窗仍在保留 CPU/RAM 预算内扩容；未知因果不会阻止该规则。压力分支实际设置每 rank 两个读槽和较小预取提交数，保留 compute pool 容量；load_case 的实体读与 Blosc 惰性数组的裁剪切片均受控，变换在槽外。重复压力不重复重建同配置；两个压力清除窗口可恢复读槽。requested_settings 与 applied_settings 分开落盘，未知分类明确为 unavailable。
- case_budget 在自然派发时读取已完成病例观察，匹配 stage、原始 shape/spacing、目标 spacing 和七文件实现身份，拒绝超过24小时、未来时间、不兼容或非完成记录。完整进程生命周期 RSS 高水位加10%仅作为保守估算下界；不把共享页减掉，也不称其为私有/增量峰值。输出字节与采用的观察路径同时记录。没有可用峰值则明确保留 header prior。活动病例的预约不低于原预约或修正后的估算；没有额外 benchmark 病例。
- 来源导出由同一 CPU 病例队列并行执行，保持来源12例、标签往返、原图 SHA、逐例 .partial→replace 和全部成功后 dataset/report 提交；汇总仍按冻结 split 顺序。中途失败保留已完成病例，不提交新的整体完成报告。

新控制回归包括真实本机四进程/两读槽、释放后下一读与已有计算并存、真实 writer 在 leader 死亡后停止、已退出/回收 leader 不发旧组信号、条件 Linux flock/身份、真实 case_budget 峰值修正与活动预约、压力 action 生效。真实 API 额外覆盖并行导出的像素/仿射/哈希/部分失败、Blosc 延迟读取的槽内切片/槽外增强和与上游逐元素一致、完整合成准备及加载/检查点恢复。

Linux `/proc` 身份、flock 和 POSIX 进程组的实际生命周期条件测试在 Windows 跳过；本机无 WSL，不安装新系统或环境。无真实 CT、CUDA/NCCL、目标服务器、端到端性能或医学效果结论。临时门控文件保留在本次 TMP，避免在可能仍持锁的慢退出进程中删除控制对象；它们不是缓存复用凭据。

最终修复后完整控制回归：**354 项，307 通过、47 条件跳过、0 失败，86.406s**，日志 `final-fixes-panel-final-green.log`。其中36项缺该控制环境的可选 Torch/Gloo/影像/API依赖，7项需真实 Linux/POSIX，4项受 Windows 链接权限限制。受影响的真实 CPU API 独立运行通过，不能把条件跳过记为通过。此前353项绿色汇总后发现临时控制目录下的失败日志引用问题，新增一项失败回归并修复路径，再执行上述最终汇总；所有此前失败和中断证据均保留在 JSON。
