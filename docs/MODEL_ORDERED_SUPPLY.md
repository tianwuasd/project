# Q 与 MedSAM2 供数边界

每个训练进程只允许当前批次构建病例。后续索引可以留在 DataLoader 中排队，读取、窗口转换和增强必须等待当前批次在全部进程消费完成。病例构建受共享并发数及病例字节估算共同限制；同一批次中，各进程的当前样本先获得名额，随后才允许较大份额的额外样本。

Q 在同步读取中保持 sample、loss 交替调用。各阶段的候选索引、完整更新、C组验证及检查点保存使用独立的主机就绪检查。准备数据或写入文件的等待发生在设备通信提交之前。全局梯度仍按原 accumulation 份额累积与归约。

MedSAM2 保留官方 DistributedSampler、BatchSampler、worker 随机种子、增强、八帧512序列与训练主循环。包装器仅给原始索引附加批次及样本位置，约束构建额度；整批数据齐备后才交给官方 Trainer。模型和初始化权重完成加载后构造 DDP，输入传输完成后执行官方步骤。消费结束会同步当前设备，再释放批次额度。官方源码保持固定提交且不修改。

主机就绪检查采用 multiprocessing Condition，逐次核对全部 rank 的阶段。该检查不使用 NCCL collective 等待数据。Q的通信超时由execution.collective_timeout_seconds设置，默认180秒；MedSAM2保留官方DistributedConf的30分钟默认值。进程失败沿用 spawn 的失败传播和已登记进程清理，阶段不一致立即终止。

## 日志

执行目录的 events.jsonl 记录以下事件，rankN.log 保存各进程输出。

| 事件 | 可核对内容 |
|---|---|
| model_cpu_wait / model_cpu_ready | rank、阶段、设备完成耗时、主机等待耗时，以及尚未提交设备通信 |
| model_build_started / model_build_completed | 阶段、批次、样本位置、实际构建并发、等待请求数、样本构建完成状态及构建耗时 |
| model_data_ready / model_data_delivered | 拼接完成后的实际就绪深度、交付后深度归零及consumer_active状态 |
| model_case_workspace | 病例、估计字节数、字节额度等待、实际预留与预算 |
| model_batch_credit_returned / model_batch_released | 全部进程消费结束及下一批额度释放 |
| model_sample_consumed | Q 的更新、样本位置及采样与计算耗时 |
| model_gradient_communication | Q 数据消费就绪后的梯度归约耗时 |
| model_forward_backward | MedSAM2 官方前向、反向及DDP通信合计耗时 |
| model_batch_consumed | MedSAM2 消费并确认设备完成的耗时 |

MedSAM2 的 nccl_wait_seconds 为 null：合计计算耗时不代表已单独测量 NCCL 内部等待。主机检查记录的等待时间可以直接核对。DataLoader 内部的索引任务数量与已构建图像缓冲分开计算；后续索引等待期间不创建后续批次的图像。

samples_built_complete表示当前批次样本已完成构建；MedSAM2此后还要执行官方collate_fn。ready_depth仅统计已经完成拼接、正在等待交付的数据，范围0至1；queue_unit在Q中为case_entry，在MedSAM2中为sequence_batch。交付后ready_depth归零，consumer_active保持1，直到当前消费阶段完成设备同步并释放额度。

## 恢复与验证

恢复核对实际模型阶段、数据集类别、当前训练名单、参数结构和完整状态。进程、批量份额及MedSAM2 worker数量保留恢复协议。训练、验证及预测使用当前实验角色，实际数组来源的几何和强度由公共读取接口提供。正常加载不额外扫描完整权重、输出或原始病例内容。

本机仅使用真实病例的短流程和CPU/Gloo。官方 MedSAM2 DataLoader 检查直接比较原始供数和有序供数的索引、图像、掩膜及帧元数据；完整训练仍受官方模型内存预算限制。CUDA设备完成、NCCL、正式网格和持续运行需在获准环境实际核验。
