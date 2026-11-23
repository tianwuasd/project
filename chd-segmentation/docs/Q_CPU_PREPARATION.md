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

`max_threads: null`允许按实际资源自动增加，可填正整数手动限制每个训练进程的准备线程数。块预算表示估算工作内存，包含原始网格halo、距离变换及临时张量。从2GiB块预算开始，按处理吞吐、可用内存和CPU额度调整，最大16GiB；可手动修改。完整区域能在当前额度和max_block_bytes内完成时直接执行一个整块；总工作量不超过64MiB的小候选/标签任务直接处理，避免线程调度开销。

所有活动块在执行前共同申请内存，已完成而尚未合并的结果仍在本批预留内；合并后释放。本组额度不足时缩减线程和块大小，最低halo块也不足时等待，取消会结束等待。已有项目100GB、整机80%及可观察的cgroup保护继续生效。并行不会使这些预算变成操作系统硬内存上限。GPU前后向、随机采样、候选顺序、训练预算、现有best/latest和缓存身份保持原有语义；`cpu_preparation`属于可调整的运行设置。ImageCHD启动器保留已有cpu_preparation及training_cache设置。

`prepare_inputs(..., parallel=runtime)`计算标签有效区、热图和监督掩膜；`pooled_image(..., use_label=False, parallel=runtime)`保持纯影像视野语义。二者不能相互替换其有效区。`with preparation(config, guard, entry, stage=...) as runtime`提供同一调度；`runtime.sections`用于带halo的非重叠区域，`runtime.map`按固定输入顺序合并结果。没有会话guard或显式禁用时使用原顺序处理路径。

日志事件`q_cpu_preparation`记录stage、case_id、完成体素/任务数、实际线程、块预算、活动工作内存峰值及调节历史。PyTorch内部`threads: 1`表示每个并行任务的内部线程数；准备并行数量看`applied_case_threads`及`peak_case_threads`。

本地实测仅ct_1003的64³和ct_1012的61×64×63区域：热图、有效区、局部输入和标签数值一致；定位器/局部模型各一次CPU参数更新并保存恢复。强制小块并行约2.09/2.11秒，整块参考约1.36/1.28秒；小块并行不具备本例速度优势。最终自动整块路径约1.40/1.18秒，对应参考1.41/1.21秒；实际并行分支另验证峰值2线程。实际额度减半时块预算315484160→157742080→315484160字节，CPU声明释放及取消通过。后续检查均为零参数更新。本机未测试完整大CT、CUDA或服务器速度。
