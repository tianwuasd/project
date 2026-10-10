# 代码阅读路线

从 README 和 workflow.md 开始。主目录的 preprocess.py/train.py/predict.py/evaluate.py 只是入口，实际实现都在 src/chd_ct/imagechd。

| 阅读顺序 | 文件 | 重点 |
|---|---|---|
| 1 | configs/chd.yaml、imagechd/config.py | 六阶段名称、输入尺寸、模型结构、学习率 |
| 2 | imagechd/common.py、preprocess.py | 七结构规则、原生/旧缓存、归一化、患者划分 |
| 3 | imagechd/dataset.py、geometry.py | 全图/区域/切片/序列样本，以及血池边界 |
| 4 | models/unet.py、grid.py、gate.py、recurrent.py | 原作者结构 U-Net、双向 ConvLSTM |
| 5 | imagechd/train.py、stage_queue.py、stage_worker.py、losses.py | 六阶段训练、忽略标签、冻结编码器和最佳权重 |
| 6 | imagechd/checkpoints.py | 完整模型集合、权重与数据划分一致性 |
| 7 | imagechd/predict.py、fusion.py | 影像独立推理、区域定位、投票、血池细化、原空间恢复 |
| 8 | imagechd/test.py、evaluate.py | 单独读取真值评估，不参与模型选择 |
| 9 | quickstart/commands.py、wizard.py | 同一功能在本机的引导与环境检查 |
| 10 | server/launcher.py、workflow.py、settings.py、jobs.py、imagechd.py、bootstrap.py | 多选计划、按需设置、后台状态与单步任务执行 |

测试 tests/test_chd_main.py 中的六阶段完整示例可作为最小使用例子；tests/test_launchers.py 检查入口分离和服务器失败处理。

要对照旧论文实现，只进入 back/paper-v1 阅读。当前主包不导入 back；原训练、推理、特征和诊断模块快照保留在 back。新增 `src/chd_ct/diagnosis/` 是当前七结构结果的独立诊断扩展，不导入 back。

U-Net 请先读 [来源与结构对齐说明](unet-source.md)，再看 unet.py 的原始主干、grid.py 的边界适配、gate.py 的可选扩展。这三部分不要混为官方网络。

服务器先看 configs/servers/zmic44.json，再按 server/launcher.py → workflow.py → settings.py → jobs.py → imagechd.py 阅读：入口解析、多选计划、有关参数、后台执行、单步骤派发。bootstrap.py 管理环境和子进程。多卡训练依赖在 imagechd/stage_queue.py；测试集入口在 imagechd/test.py。tests/test_server_jobs.py 覆盖产物衔接与状态；test_server_posix.py 用真实 Linux 进程验证后台存活、锁、失败和中断。

诊断阅读顺序：`labels.py` 看空白为阴性的来源记录，`features.py` 看 56 个结构描述，`classifier.py` 看训练及真实分支解释，`predict.py` 看 HTML 输出，`evaluate.py` 看独立评估。候选规则在 `rules.py` 与 `configs/diagnosis-rules.yaml`。运行方法见 [诊断说明](diagnosis.md)。
