# 剩余病例自适应加速实施计划

> **For agentic workers:** 按下列任务执行并独立复核；公共资源与数值流程由主agent负责，训练供数由指定子agent实现后交主agent复核。

**Goal:** 让剩余少数病例使用安全的病例内并行和动态块，并消除训练供数的无效同步与资源闲置。

**Architecture:** 基于公开main62a453012b33e89e27799127fbe391334d8db186，保留其他模型变更。新增预处理执行策略模块，分离CPU登记、动态块与数值函数；训练使用最多一个顺序预取批和有界缓冲。

**Tech Stack:** Python、NumPy、SciPy、MONAI、PyTorch、concurrent.futures、既有MemoryGuard。

**Spec:** ../specs/2026-11-16-adaptive-blocks-design.md

## Global Constraints

- 正式有效内存目标最多80%，项目合计最多100,000,000,000字节；保留实时测量、共享登记及压力等待。
- 冻结计划、配方、数据划分、标签语义、global batch、patch、epoch及RNG消费顺序保持。
- 代码在D:\projects\project\chd-segmentation，测试在已说明的back目录；不连接服务器。
- 修改均用apply_patch；未知错误明确失败。只发布本项目源码，不上传数据、权重或环境。

## Review Focus

- 输出区间增大/缩小不能遗漏或重复体素，边界与类别并列必须与原实现一致。
- 预滤波保留完整轴线，各轴之间有完成屏障。
- 剩余一个病例时CPU/RAM预留按实际执行量记账，多个任务不能重复扩大额度。
- 暂停、取消和未知异常要释放本任务线程、票据、映射及预取，保护完整病例和原件。
- CUDA pinned源生命周期、跨rank停止、预取重放和检查点恢复不能改变。

## Tasks

- [x] 更新本地到固定main内容并核对逐文件blob，保留整合后的模型与等待接口。
- [x] 新增preprocessing/scheduling.py：登记实际活动病例、分配病例内CPU份额、清理退出的成员登记。
- [x] 新增models/stunet/block_execution.py：有界线程波次、真实RAM准入、自适应区间长度、吞吐调整、实际进度与耗时。
- [x] 修改streaming.py/cache调用：正常保留映射缓存，压力才回收；图像与标签插值共享确定性全局坐标；预滤波并行独立完整轴线。
- [x] 修改budget.py/config.py：仅按待处理病例规划病例进程，公开块/线程上限与调整窗口，并为训练顺序预取增加真实内存额度。
- [x] 修改supply.py/engine.py/stage_memory.py：有界下一批预取，安全异步传输，降低nvidia-smi频率，减少重复规划。
- [x] 编写维护型真实短测入口；得到用户授权后验证两例、最多两次更新、精确恢复与自动清理，记录实测边界。
- [x] 审查准确源码兼容映射并准备独立服务器更新/预处理/训练入口；发布时以固定main基线和完整文件摘要核验本项目及其他仓库根目录。

## 当前验证状态

141个Python文件通过AST语法解析，git diff --check通过。两例真实ImageCHD的四次数值比较完全一致；病例内最高4线程、块256KiB至8MiB；实际预留额度收紧时块降至约1.30MiB，释放后恢复8MiB，输出完全一致。两个CPU/Gloo进程共两次更新及精确恢复通过。工作目录已自动清理，保留回执在批准的2MiB内。完整验收范围见STUNET_ADAPTIVE_VALIDATION_20261116.md。精确源码兼容映射已更新，发布指针及文件核验结果由GitHub main与本地同步回执记录。
