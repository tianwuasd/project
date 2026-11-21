# Q-r2定位器内存与资源查询修复

## 目标与边界

沿用现有预处理、A/B/C/D名单、完整视野、毫米距离、四倍池化、网络参数、损失和完整检查点。定位器只改变输入及监督的构造方式：CPU按块读取原网格，监督在原网格计算后池化。分组归一化仍处理整个粗网格。原基座40000次更新保持可用，继续任务直接进入localizer。

项目100GB、整机80%、系统预留、同行未兑现预留和实际内存监控继续生效。额外50%规划缓冲优先保留；只有原始最小布局能够容纳时允许收缩该缓冲。原始需求超过当前余量时等待，超过所有允许布局的硬上限时明确终止并报告。

## 实现

- 定位器新增独立的粗网格入口，旧forward的padding/pooling定义保持不变。CPU逐块生成相同的平均池化图像、有效掩膜与监督。监督块按8mm半径加足够的原网格halo，保持原距离变换、可靠区域和最大池化语义。禁止先降采样标签再计算距离。
- 新的构造函数对完整网格进行遍历，保存粗网格结果；及时释放源映射驻留页。CUDA训练使用PyTorch保存反向张量到CPU的正式接口，减少同时驻留显存的激活；CPU额度包含粗网格及保存的反向张量。两例短测必须比较原流程的监督、损失与参数梯度。
- 内存profile明确区分原网格块、粗网格输入/监督、反向张量、模型与检查点。定位器不再套用完整多类预测数组除以5的估算，也不把整个映射文件尺寸当作常驻副本。
- 公共limits提供额外缓冲与原始容量函数；planning、弹性扩容、验证预算复用。字段区分raw_minimum_bytes、estimated_ram_bytes、requested_memory_margin_fraction、memory_margin_fraction、margin_bytes。Q3两个消费者由助手2在接口固定后适配。
- runtime对允许的所有拓扑计算硬上限下的原始minimum；不可能准入时报告具体需求和上限。暂时资源不足保留取消与显式等待时限。
- 显卡查询设置默认30秒超时、最多2次额外重试、间隔2秒。仅重试确切TimeoutExpired，其他错误直接传出。显卡表与计算进程表在同一批次重试，失败时丢弃部分结果；重试日志记录原因、次数、耗时和取消。

## 调用面

models/q_route1/localizer.py、adapter.py、新增training_inputs.py；models/memory.py；models/workflow.py仅连接guard；acceleration/limits.py、planning.py、runtime.py、elastic_memory.py；models/q_route1/validation_runtime.py；显卡查询涉及config.py、registry.py和measurements.py。

## 验证与交付

- [ ] 只读核对服务器实际配置与逐例shape/locations/array字节元数据，重算284402014104B；元数据缺失时明确范围。
- [x] 原始minimum、额外缓冲、暂时不足、硬上限、取消及显卡查询配置的真实小检查。
- [x] 两例真实病例的小区域完成输入、监督、损失、全部梯度对照；双CPU一次全局更新、best/latest与单独获准的零更新恢复完成，不复制原数据。
- [x] 记录单进程内存峰值、文件数量、大小和时间；服务器CUDA/NCCL及全尺寸容量单列未验证范围。
- [ ] 独立提交交秘书审核，不并发推main；审核发布后提供固定SHA的--force更新和跳过base的localizer继续命令。

当前缺少服务器localizer.json、prepared.json内逐病例storage元数据；不能据现有日志确认产生最大项的病例及精确新容量。

## 已验证范围

ct_1003与ct_1012原区域≤64³，技术网格[64,33,33]及[63,33,33]。原输入、valid、heat、mask、损失与全部参数梯度最大差异均0；第三任务无可靠阳性，覆盖受限。双CPU/Gloo执行一次全局更新，两份参数一致，best/latest写出；单独获准的零更新恢复restored_exact=true、update仍1，模型/优化器及逐rank随机状态恢复通过。实际优化器调用共2次，临时产物自动清理。

CUDA反向卸载、Linux/NCCL和完整病例容量未实测。原始minimum超过动态预算时曾真实等待，额度恢复后以收缩的额外缓冲准入，80%与100GB未关闭。正式任务直接启动localizer；有localizer/latest.pt才使用resume，不重复base更新。
