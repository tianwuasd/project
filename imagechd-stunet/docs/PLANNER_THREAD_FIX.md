# v0.3.4：预处理规划线程修复

用户在2026-10-26提供的日志显示：12例来源病例已导出，8例fit指纹提取完成，生成网络规划时触发 `RuntimeError: set_num_threads expects a positive integer`。没有完成病例预处理，也没有进入完整尺寸GPU预检、正式训练或预测。

## 原因与修复

v0.3.2把通用 `configure_paths()` 中的 `nnUNet_n_proc_DA` 从2改成0，用来保持训练串行增强。但nnU-Net 2.8.1规划器也读取这个变量，并直接把它交给 `torch.set_num_threads()`；计算线程数必须为正数。这是项目改动引入的回归，不能归因于服务器显存不足。

现在通用准备设置使用 `max(1, WHOLE_HEART_CPU_THREADS)`：受管规划遵循共享CPU预算，单病例子进程遵循其1线程预算，未配置时用1。正式训练、完整尺寸短测和预测入口仍明确设增强进程数为0；不改变训练预算、划分、模型或新任务20 GiB规划预算。`0`个增强子进程不等于`0`个计算线程。

依据：[官方2.8.1规划器](https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py)、[线程设置读取函数](https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/nnunetv2/utilities/default_n_proc_DA.py)。具体方法摘要和本机依赖版本见 [验证记录](PLANNER_THREAD_VALIDATION.json)。

## 已验证的范围

- 新增5项线程回归：先复现旧实现失败，再验证共享预算、单病例预算、默认值和串行增强边界。
- 官方规划估算方法体配合真实Torch和动态3D U-Net：复现0线程异常，修复后1/4线程均返回相同的正估算值，并恢复调用前的Torch线程数。
- 现有检查点恢复CPU验证再次通过：权重、优化器、随机状态、恢复后一步训练以及旧无RNG字段检查点。
- 面板回归104项：99通过、5项因Linux或影像依赖条件跳过。

这不是完整nnU-Net规划、真实CT预处理或GPU验收。本机Torch为2.12.1+cu126，服务器固定Torch2.8.0+cu128和nnU-Net2.8.1仍须实测。本轮按用户要求没有新增全流程快速测试模式。

## 更新后如何处理失败任务

1. 在原runtime用菜单`u`更新并重开；旧更新入口不可用时按 [手动Git引导升级](SERVER_GIT_UPDATE.md)。标题应为v0.3.4或更新。
2. 用`n`新建任务，包含预处理以及本次需要的后续步骤，核对B0、ct_holdG、20 GiB和获准GPU范围。
3. 本次失败任务的预处理尚未完成，不能当作完整缓存复用；选择它为来源并同时选择预处理时，程序会提示并独立新建缓存。修复改变预处理实现签名，旧版本缓存也必须重新核对，不能强制改写签名。
4. 原任务`r`使用原冻结源码，仍会执行旧bug；更新不会热改旧任务快照。本补丁没有原地迁移冻结任务的功能，也不篡改`plan.json`或`code_manifest.json`。

原结果和失败日志保留。服务器实际部署、真实数据预处理、GPU预检与训练通过情况，以服务器后续输出为准。
