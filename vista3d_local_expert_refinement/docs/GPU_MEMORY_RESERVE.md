# GPU容量安全余量（面板0.5.3）

用户于2026-10-28确认：当前VISTA协同训练的显存安全余量下限由2 GiB改为4 GiB。
规划上限仍为20 GiB，不改预处理的patch/spacing、250轮×250训练/50验证或模型方法。

每张卡重新读取总显存和当前空闲显存，按字节计算：

```text
reserve = max(4 GiB, physical_total × 10%)
usable = max(0, min(planner_budget, fresh_free − reserve))
```

| 总显存 | 当前空闲 | 规划上限 | 预留 | 用于容量估算的usable |
|---|---|---|---|---|
| 24 GiB | 23 GiB | 20 GiB | 4 GiB | 19 GiB |
| 24 GiB | 24 GiB | 20 GiB | 4 GiB | 20 GiB |
| 24 GiB | 23 GiB | 16 GiB | 4 GiB | 16 GiB |
| 64 GiB | 63 GiB | 60 GiB | 6.4 GiB | 56.6 GiB |

余量只从现场空闲量扣一次，不能再从20 GiB规划上限扣4 GiB变成固定16 GiB。
`gpu_budget.py`集中保存下限与比例；`cooperative_control.admit`生成每卡usable，
`training_execution.capacity_batch`利用已初始化优化器的完整尺寸allocated峰值推算batch，
所有rank采用可共同承受的每卡batch，再实际确认。证据不全保留已证明batch，
不把估算当作硬性分配或长时间稳定性保证。

阶段决策JSON增加`gpu_memory_reservation`，记录minimum_gib、fraction_of_total、
每卡usable_bytes及计算范围；capacity_inference的usable_scope同步写4 GiB规则。
reserved峰值单独观察，不能将整个缓存再从usable中重复扣除。CUDA上下文和NCCL等
非Torch分配也会占显存，因此4 GiB是预算余量，不保证运行中始终空闲4 GiB。

初始空闲准入门槛、外部PID排除、可见性保护和0.5.2阶段切换复查保持原逻辑；
准入只决定是否可进入实际容量验证，不代表20 GiB预算已全部可用。
预测仍为按病例分卡的滑窗推理，不据此声称预测受4 GiB硬限制。

已识别训练GPU OOM在有界恢复预算内将每卡batch减半，整组退出后重验并恢复最近完整
epoch检查点；当前恢复配置最多3次缩小，前置容量恢复已用次数计入。batch=1仍失败、
恢复耗尽、未知/数据/CPU错误则停止并保留证据。没有完整检查点时只能从seed0重新开始。
每个完整epoch保存检查点；失败以来未提交的更新会丢弃，不继续使用可能只更新了一半的状态。

更新后新建训练可继续选择兼容的既有预处理。本改动不改依赖和七文件准备身份，旧冻结
任务或正在运行的任务不会热更新；需要新建任务才能使用0.5.3规则。Skill v2随包源码、
ZIP与逐文件清单从已发布固定项目提交重建，两类v1与独立CT示例保持原样。

本机回归覆盖4 GiB、较大卡10%、不重复扣除、batch推算、决策落盘和原调度恢复。
目标Linux/CUDA/NCCL、真实Wholeheart及长期显存曲线未在本次操作中运行。
