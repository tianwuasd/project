# Q定位器与局部精修CPU准备

配置`cpu_preparation`控制热图、局部定位影像池化、候选监督检查和局部标签构造。默认启用，复用预处理的`BlockExecution`调度，CPU任务内核保持单线程；CPU处理块通过线程池并行。不同GPU进程分别准备输入，CPU额度在项目`resources/active.json`内按真实进程身份共享登记，块计算不持有资源文件锁，阶段结束归还额外CPU额度。线程上限受有效CPU核数×cpu_fraction、其他登记任务及本组份额限制。

```yaml
cpu_preparation:
  enabled: true
  max_threads: null
  block_bytes: 2147483648
  max_block_bytes: 17179869184
  growth_seconds: 2.0
```

`max_threads: null`允许按实际资源自动增加，可填正整数手动限制每个训练进程的准备线程数。块预算表示估算工作内存，包含原始网格halo、距离变换及临时张量。从2GiB块预算开始，按处理吞吐、可用内存和CPU额度调整，最大16GiB；可手动修改。完整区域能在当前额度和max_block_bytes内完成时直接执行一个整块；总工作量不超过64MiB的小候选/标签任务直接处理，避免线程调度开销。分块时先按当前块预算与一半实时余量确定单块，再以剩余额度限制并发；请求更多线程不会压小已选单块。完整候选任务保留其工作区大小，仅调整并发数量。吞吐下降时即使实际线程数已经达到上限，也会减少并发；后续增长受实际CPU和内存上限约束。

所有活动块在执行前共同申请内存，已完成而尚未合并的结果仍在本批预留内；合并后释放。本组额度不足时缩减线程和块大小，最低halo块也不足时等待，取消会结束等待。已有项目100GB、整机80%及可观察的cgroup保护继续生效。并行不会使这些预算变成操作系统硬内存上限。GPU前后向、随机采样、候选顺序、训练预算、现有best/latest和缓存身份保持原有语义；`cpu_preparation`属于可调整的运行设置。ImageCHD启动器保留已有cpu_preparation及training_cache设置。

`prepare_inputs(..., parallel=runtime)`计算标签有效区、热图和监督掩膜；`pooled_image(..., use_label=False, parallel=runtime)`保持纯影像视野语义。二者不能相互替换其有效区。`with preparation(config, guard, entry, stage=...) as runtime`提供同一调度；`runtime.sections`用于带halo的非重叠区域，`runtime.map`按固定输入顺序合并结果。没有会话guard或显式禁用时使用原顺序处理路径。

日志事件`q_cpu_preparation`记录stage、case_id、完成体素/任务数、实际线程、块预算、活动工作内存峰值及调节历史。PyTorch内部`threads: 1`表示每个并行任务的内部线程数；准备并行数量看`applied_case_threads`及`peak_case_threads`。

## 局部阶段调用范围

- 候选准备：`QAdapter.prepare_training → frozen → locate → pooled_image`通过公共`sections`准备定位输入；冻结基座概率、定位特征与候选由现有`TrainingCache`按上游权重和输入身份复用，连续使用同一病例时复用驻留结果。
- 候选监督：`prepare_training → preparation.map → crop(parallel=False)`对独立候选并行构建输入和标签，按原候选顺序合并；每个候选有独立的完整工作区，资源不足只减少同时处理的候选数量。
- 局部训练：`sample → frozen → crop → allowed`保留分层抽样、随机扰动和局部模型前后向。标签支持集合使用公共`map`，小任务直接顺序计算。每次抽样的局部输入仍按选中的候选构建，随机扰动结果不进入确定性缓存。
- 局部验证：`validate_parallel → validate_records → local_probability`使用`ValidationWorkspace`和`LayerInference`按验证额度执行，未调用`BlockExecution`的并发增长逻辑。候选选择与局部融合按原顺序执行。
- 局部预测：`predict → locate → pooled_image`使用公共`sections`；局部候选、模型计算和融合保持顺序。预测的池化有效区来自影像视野，不读取标签构造监督mask。

公共`BlockExecution._capacity(block_bytes=...)`支持先确定单块预算再限制并发；原无参调用保留其块预算规则。资源登记继续保留训练进程、供数进程和预取的基础CPU额度。块计算不持有全局资源锁。DDP在各进程完成当前输入和计算后才进入梯度同步；公共CPU调度无法消除病例自身处理量差异。

本地实测仅ct_1003的64³和ct_1012的61×64×63区域：热图、影像视野、监督mask、局部输入和标签数值一致；定位器/局部模型各一次CPU参数更新并保存恢复。自动整块路径约2.42/2.16秒，参考约2.27/2.02秒；手动小块路径为4块、约4.18/4.00秒，本例未证明持续提速。支持集合计算另验证峰值2线程；真实额度检查中1线程和2线程的单块预算均为331109376字节，临时占用额度后缩小至165554688字节，释放后恢复。CPU基础额度及活动声明在重新规划后保留，声明释放、缓存复用及取消通过。通过流程耗时52.90秒，进程内存峰值1225109504字节；本机未测试完整大CT、CUDA、多卡通信及服务器速度。
