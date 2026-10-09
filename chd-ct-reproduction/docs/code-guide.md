# 代码阅读路线

从 README 和 workflow.md 开始。主目录的 preprocess.py/train.py/predict.py/evaluate.py 只是入口，实际实现都在 src/chd_ct/imagechd。

| 阅读顺序 | 文件 | 重点 |
|---|---|---|
| 1 | configs/chd.yaml、imagechd/config.py | 六阶段名称、输入尺寸、模型结构、学习率 |
| 2 | imagechd/common.py、preprocess.py | 七结构规则、原生/旧缓存、归一化、患者划分 |
| 3 | imagechd/dataset.py、geometry.py | 全图/区域/切片/序列样本，以及血池边界 |
| 4 | models/unet.py、recurrent.py | 复用的 U-Net、双向 ConvLSTM |
| 5 | imagechd/train.py、losses.py | 六阶段训练、忽略标签、冻结编码器和最佳权重 |
| 6 | imagechd/checkpoints.py | 完整模型集合、权重与数据划分一致性 |
| 7 | imagechd/predict.py、fusion.py | 影像独立推理、区域定位、投票、血池细化、原空间恢复 |
| 8 | imagechd/evaluate.py | 单独读取真值评估，不参与模型选择 |
| 9 | quickstart/commands.py、wizard.py | 同一功能在本机的引导与环境检查 |
| 10 | server/launcher.py、imagechd.py、bootstrap.py | GPU/环境/目录设置和对应任务启动 |

测试 tests/test_chd_main.py 中的六阶段完整示例可作为最小使用例子；tests/test_launchers.py 检查入口分离和服务器失败处理。

要对照旧论文实现，只进入 back/paper-v1 阅读。当前主包不导入 back；原训练、推理、特征和诊断模块已从主包移出，不存在两个并列默认入口。
