# CPU调度、进度与恢复

来源固定为server-engineering VISTA0.7.2，commit `47016371e8d56a19630585f49c2cc0299259fa34`。`vendor/vista_live_resources.py`保留该版本的CPU列表解析和Linux实时快照函数；没有复制训练、DDP或GPU调度模块。可追溯来源：[server-engineering skill](https://github.com/tianwuasd/codex-skill/tree/main/server-engineering)。

采用：获准CPU与实时空闲时间份额的80%预算；阶段RAM为可用量80%与保留至少4GiB后的较小值；阶段总预算与新鲜余量分别约束准入。Windows核对CPU affinity和psutil实时数据，Linux同时读取当前及祖先cgroup quota/cpuset/memory限制。关键资源未知时不放行大体积病例。

每个病例用独立子进程，数值库线程固定1；默认最多2个进程。进程槽同时约束读盘和计算，因此重读并发不会超过2，没有后台无界预取。未来增加流水阶段时需要另建有界读取闸门，不直接放大进程数。

阶段上限扣所有活动病例预留；实时可用内存已经扣掉驻留部分，因此第二层只扣仍未兑现的预留。只对本次进程树中已核验身份的私有USS给予抵扣；采集失败不猜测可释放容量。Windows虚拟环境启动器的计算子进程也纳入统计和清理。

影像以小z块统计，结束后释放，再读取标签。标签连通域在每类包围盒内计算，组件大小按平面累计；估算包含完整标签、掩膜、最坏连通域缓冲、计数和固定开销。资源预算是准入策略，不是操作系统硬内存限额，也不能保证拦住突发外部OOM。

父进程持续检查资源；主机余量低于保留线时只停止本次拥有的进程，保留已完成项，缩小后续worker/RAM预算并有限重试，最多3次。无法准入时有界等待后停止并留状态，用户可释放资源后resume；不终止外部任务。

每次运行有provenance、resolved_config、冻结code_snapshot、events.jsonl、run.log和status.json。事件记录requested/applied、准入原因、峰值、耗时和压力调整。旧任务的worker使用原冻结代码；resume校验冻结代码和输入合同，不以当前Git HEAD冒充原版本。

操作互斥使用操作系统锁，进程退出后由内核释放；进程身份JSON原子提交，损坏的半写记录可在持有OS锁后恢复。父进程持久记录worker出生身份，worker使用父进程身份与随机nonce进行自监护；父进程被强制结束时worker退出。接管前核对仍存活的所属进程，不仅检查父PID。

报告也取得项目操作锁和资源预算。三平面QC的影像、标签都按块提取切面，不生成整卷float64标签。每次报告写入独立report_versions目录，完整成功后才切换report-current.json及稳定HTML入口；失败不会覆盖先前版本。

恢复会重新读取已完成病例的当前文件SHA256，可能需要额外I/O。仅size/mtime不构成全字节验证。失败病例重算，未完整提交病例重算；没有按文件名直接复用的捷径。

`smoke --profile capacity`包括最大第三维、影像dtype与标签dtype代表；它验证完整正式链路，没有裁小真实病例。它不构成持续加速收益证明，默认并发仍由实时CPU/RAM和I/O共同决定。
