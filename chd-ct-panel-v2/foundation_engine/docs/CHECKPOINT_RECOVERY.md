# v0.3.2 检查点恢复与20 GiB新计划

2026-10-26用户日志显示，B0完整尺寸短测完成了训练/验证并写出损失，随后在恢复检查点时失败；正式训练与预测未开始。原因是项目把已加载的dict交给nnU-Net 2.8.1父类，父类实现仅在str分支初始化内部checkpoint变量，触发`UnboundLocalError`。这段日志没有显示OOM，增大显存预算本身不能修复该异常。

修复让父类始终接收文件名，Path转换为str；兼容dict时使用临时文件并在成功/失败后清理。保留原子保存、权重/优化器和项目RNG恢复，旧无RNG字段检查点仍能加载。使用的是 [nnU-Net v2.8.1官方实现](https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py) 的真实保存/恢复方法体，实际CPU验证见`REPAIR_VALIDATION.json`。

## 服务器操作

1. 先修通服务器代理出口；现有SSH部署密钥的公钥指纹已由用户与GitHub页面比对一致，页面为Read/write。节点连接超时与密钥匹配是不同问题。
2. 在原runtime使用面板`u`或`bash start_server.sh --update`，按提示重开，标题应为v0.3.2。旧更新器不可用时用 [直接Git升级](SERVER_GIT_UPDATE.md)。本项目原runtime为`/data5/zhougaowei/zhangruichen_workspace/vista3d_runtime`，不能换目录绕过任务锁。
3. 使用`n`新建任务，选择`environment,prepare,smoke,train,predict,summarize`中本次需要的步骤；本次从旧5 GiB改为20 GiB，必须包含prepare。可选择旧任务作为来源，程序会提示并新建计划/预处理缓存；不修改旧任务、旧原始数据或旧结果。
4. 新任务预览核对原始数据、runtime、结果根、模型B0、ct_holdG、GPU候选和并发上限、20 GiB预算。只使用实际获准的卡；空闲探测不是预约。
5. 查看完整尺寸预检日志，确认检查点恢复及后续阶段实际通过；提交成功和短测loss不代表正式训练完成。

旧任务`r`仍使用原冻结源码，更新面板不会修补其快照；不要篡改`code_manifest.json`或`plan.json`绕过完整性校验。本版本未提供将旧冻结5 GiB任务迁移到修复代码的原地续跑功能；本次新20 GiB实验从新计划开始，之前没有已完成的正式训练需要接续。

## 显存含义与兼容边界

用户最新指定新任务预算20 GiB，替代先前24 GiB要求。历史硬件记录中RTX3090每卡24 GiB，因此名义上留4 GiB；这是规划预算与总量之差，不能当作实际峰值余量或无OOM保证。该参数可能改变patch/batch/网络计划，不是硬性显存限额。

新默认覆盖旧无策略版本的5 GiB保存偏好；本次显式参数和新策略下保存的自定义预算优先。仅预测/查看旧模型沿用原计划。新计划和预处理缓存独立，不能把20 GiB计划用于旧5 GiB模型续训。

等待与启动按该卡实测总显存计算空闲门槛：`min(max(6000, (预算GiB+0.5)*1024), floor(总MiB*0.95))`。20 GiB、24576 MiB卡门槛为20992 MiB；保持计算进程、利用率和分配范围检查。预算超过容量立即报错；显式满卡预算也不再要求不可能的24.5 GiB空闲。实际CUDA完整尺寸预检仍需目标服务器运行。

CPU检查使用本机Torch2.12.1，服务器依赖仍固定Torch2.8.0+cu128与nnU-Net2.8.1。本机验证不替代这组服务器依赖、Linux进程边界、真实CT/GPU及端到端训练验收。原始影像、权重、私钥和代理订阅均未加入Git。
