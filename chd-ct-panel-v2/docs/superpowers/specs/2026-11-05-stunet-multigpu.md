# STU-Net-S V2 多 GPU 适配

用户已选择 server-engineering V2 策略并要求继续实现。沿用当前 CLI、官方模型与 partial-MYO 监督；不新增面板、不合并到预处理 main、不启动服务器正式训练。

## 合同

- 固定全局 batch 默认 4、128³、1000 epochs、每轮 250 train attempts/50 validation attempts。卡数最多 min(batch, 获准可见卡数)，份额为 [4]/[2,2]/[2,1,1]/[1,1,1,1]。容量不足不改变 batch/patch。
- 一个 CUDA rank 对应一个物理 UUID；继承 CUDA_VISIBLE_DEVICES，仅使用可靠 PID 证据为空闲的卡。Linux 多卡使用 NCCL，CPU Gloo 仅为数值验收。
- Partial CE 先全局累加有效体素数与损失分子，Dice 先在每个样本上屏蔽缺失类别再全局归约统计。归约支持 autograd，配合 DDP 的梯度平均，不能平均各 rank 的局部 loss。
- 全局槽编号决定前景规则和 patch RNG，成员数只改变槽的归属。不同成员布局不承诺浮点逐位一致。
- CPU/RAM 80% 为全组总预算，扣除监督器、rank、串行取块/驻留输入后分配持久供数 worker。workers 参数是全组上限；每 worker 内层一线程，prefetch 至少等于 workers，读并发另限。
- 每 rank 真实 local batch 前反向和优化器动量容量准入。AMP 任意 rank 溢出，全组同步跳步，尝试仍消费，不补样本。
- 原子完整检查点包含模型、SGD、scaler、游标、逐 rank RNG 和成员身份。中途恢复须保持成员；成员只在完整 epoch 边界变化。旧冻结代码不热修，不放宽任何旧签名。
- 自动扩卡仅完整 epoch；同工作量扩卡首轮暖机后连续三轮慢于原组末三轮中位数 25%，才回原组；原组至少四轮，否则不作退化结论。新增卡在本阶段延期。
- 主机压力最多三次缩减全组供数预算并从最新完整 epoch 恢复。清理仅限本监督器启动且出生身份匹配的进程，不清理全局 SHM。
- 日志记录 requested/applied、UUID、布局、供数与等待、尝试/更新/溢出、完整 epoch 墙钟和重组原因。测试范围与目标 Linux/NCCL 实测分开。

## 来源

server-engineering V2 0.7.2：references/vista-v2-performance.md，以及 bundled batch_distribution.py、uneven_batch_loss.py、training_performance.py。资源/归约机制按 STU-Net partial-MYO/ignore8 重新适配，不能直接复制 B0 loss。工程结构依据 engineering-project-reference。
