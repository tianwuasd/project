# 0.6.1：固定上限8、可工作的CPU供数和扩卡退化恢复

本次用户确认：全局batch最多8。达到8后新增卡拆分同8块，不继续增长到10/12。
当前默认全来源30训练/8验证/20独立G测试、原生随机病例/前景取块、250轮、每轮250训练尝试和50验证保持。
容量不足可降低batch并明确记录；不是保证单卡也能容纳8。新任务用新代码，旧冻结任务不热修。
0.6.0兼容预处理身份未改变，可选择该版本的来源任务；旧8/4预处理仍不兼容。

## 直接使用

在独立工程目录执行 `python -B tools/quick_check.py` 可获得简短JSON基础结果；
完整控制用 `python -B tests/run_panel_suite.py`。均不安装环境、不提交训练、不读取真实CT。
可选依赖缺失如实跳过；CPU API与目标Linux CUDA/NCCL验收另计。

## batch及损失

`batch_distribution.layout` 将已证明每卡容量乘卡数后封顶8，再作非空均衡划分：
4卡[2,2,2,2]，6卡[2,2,1,1,1,1]，8卡每卡1。8个块最多使用8张训练卡。
`cooperative_control` 将真实总量/逐卡量传入容量、正式rank、资源估算和日志。
逐卡payload/private内存及CPU/RAM份额随实际batch分配，不能全按较大rank估计。

nnU-Net2.8.1原生全局前景oversampling保持。不能给每rank独立取整而改变全局比例。
不均分时，`uneven_batch_loss` 给CE乘 `world*local/global`，使DDP平均梯度仍按块平均；
batch Dice先把本rank的intersect/pred/ground按batch求和，再用AllGatherGrad聚合等尺寸类别统计。
这避免原按[local_batch,classes]收集在不同batch下的形状错误。深监督权重保持。
此适配限当前B0类标签、无ignore区域合同；未适配的损失需明确处理，不能盲目套权重。
相等分配保留上游损失实现。成员变化仍不宣称逐位相同的随机轨迹或保证线性提速。

## CPU、队列与RAM联合分配

`live_resources._rank_setting` 同时保留训练/验证驻留进程、main/pinned输入和队列，
要求 train_prefetch>=train_workers、val_prefetch>=val_workers；有余量时取约2倍worker。
RAM紧张时优先减少同时驻留但在训练阶段闲置的验证worker，再确定可工作的训练worker。
反馈扩池使用相同约束，不会从12个能工作进程扩成13个进程/3个槽。
CPU/RAM80%预算、实际threadpool限制、精确已消费游标与未消费重放保持。
队列能派发不代表CPU必然80%忙；要看实际产出PID、等待、压力及吞吐。

## 扩卡退化恢复

不比较一长串候选。使用原组自然完成的最后3轮中位耗时，要求此前至少4轮（留出暖机）。
只有全局batch相同、同冻结计划和250/50计算预算的扩卡才可比较。
新组首轮暖机不判退化；随后连续3个完整epoch均比基准慢25%以上时，触发完整epoch边界重组。
继续最新已保存检查点，保留扩卡期间已完成轮次，不回滚权重进度。
旧组重新做现场容量/所有权检查；新增卡对本阶段延期，避免10分钟后反复加入。
跨连续扩容保留原可比基准；不足数据或batch改变则记录不可比，不编造加速结论。
25%/3轮/1暖机是本版工程判定值，不是论文通用参数。

## 日志和编译

`feedback.rankN.json`为有界最新快照，`feedback.rankN.jsonl`追加窗口/变更证据；
完整CPU拓扑和初次资源证据在stage-decisions，窗口不重复写入。
内存只保留最近两个反馈窗口/事件；面板兼容读取旧数组和新schema3快照。
`epoch-performance.jsonl`记录完整轮次的逐rank墙钟与扩卡判断；这些时间包含验证、保存、反馈，
不能当作纯GPU内核时间，也不能与各分段重复相加。

旧编译检查把全部logits相加，会人为放大半精度梯度。新检查对每个输出最多均匀抽1024个值，
用归一化投影验证梯度，同时继续比较完整输出、形状、dtype、有限性及原误差界。
不修改训练损失；恢复RNG/缓冲/梯度，不更新优化器。任一rank失败仍全组回退，
CPU检查通过不能声称目标CUDA编译已成功。

## 必须保留的检查

- test_capped_batch：4/5/6/7/8卡上限、逐卡内存、实际6卡13/3缺陷回归。
- test_batch_pool：RAM限制后的每个预约进程实际接到任务，以及原重放/清理。
- capped_batch_gloo_smoke：实际CPU Gloo四/六rank，Dice+CE+深监督梯度/更新对照未拆分8块。
- test_performance_repair：持续/短暂/不可比退化、最新完整epoch恢复、日志有界追加。
- test_execution_transfer：大输出half探针不人为溢出，偏置/非有限/结构/梯度不一致仍拒绝。

详见CAPPED_BATCH_VALIDATION.json。没有在服务器执行真实CT或CUDA/NCCL训练；实际速度需用新任务短测验收。
