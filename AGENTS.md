# 先心代码助手2

本目录负责 MedSAM2 与 Q-20261108-路线一，版本 q-route1-20261111-r2。

- 工作分支为 `codex/chd-segmentation-assistant2-medsam2-q-r2`，基点为 `298a75a50e18117bd62c741ebbc328b89dc8e7ad`。
- 所有编辑、暂存与提交在本工作目录执行。原目录 `D:/projects/project/chd-segmentation` 由先心代码助手1维护。
- 保留本目录自己的 Git index；不切换原目录分支，不暂存原目录内容，不调用项目 update.py，不合并 main。
- 共享训练调度、GPU分配、CPU供数及正式数据划分由助手1维护。本分支通过模型独立入口与已有准备描述接入。
- Q 保留 r2 的类别、网络、损失、求解和一次整例融合。预处理输入沿用已有准备描述；不改变标签语义或患者分组。
- MedSAM2 使用官方序列训练及记忆传播；提示来源分别登记。预测过程中禁止读取测试标签生成提示。
- 本次已获准在 `D:/projects/back/t17/chd-segmentation-assistant2` 使用最多2例真实病例，每阶段最多2次更新，最多200个验证文件、峰值8 GiB，结束清理副产物。验收报告保留在 `D:/projects/back/sync-reports/chd-segmentation-assistant2`，不超过10 MiB。
- 已获准将固定提交的 MedSAM2 官方源码保存在 `vendor/MedSAM2`、独立Python依赖保存在 `env/medsam2`，合计最多1 GiB、约1万个文件，保留复用。权重下载需要核对大小和授权。
- 原始病例、个人路径配置、验证产物、环境和权重不提交。
