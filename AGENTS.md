# 先心代码助手2

本目录负责 MedSAM2 与 Q-20261108-路线一，版本 q-route1-20261111-r2。

- 工作分支为 `codex/chd-segmentation-assistant2-medsam2-q-r2`，基点为 `298a75a50e18117bd62c741ebbc328b89dc8e7ad`。
- 所有编辑、暂存与提交在本工作目录执行。原目录 `D:/projects/project/chd-segmentation` 由先心代码助手1维护。
- 保留本目录自己的 Git index；不切换原目录分支，不暂存原目录内容，不调用项目 update.py。用户已要求继承助手1已发布main：远端6ef3fd4c51eb53da26faa54a147fc35ae15ebc40的chd-segmentation子树与本地助手1提交c28e191逐文件相同。
- 共享GPU分配、资源登记和CPU预算沿用助手1的training/resources.py。MedSAM2和Q分别接入模型入口，保留各自训练方法、预算和数据分组。
- Q保留r2的类别、网络、损失、求解和一次整例融合。用户已要求基于原1282冻结划分生成Q组合：仅原训练患者按r2的A:B=2:1进一步分组，C保留原验证，D保留原内部及外部测试身份；统计仅在A拟合。
- MedSAM2使用官方序列训练及记忆传播。用户已授权标签自动生成框的正式评测；该入口明确登记标签访问及框生成规则。普通无标签预测继续只接受人工或图像定位器提示。
- 本次已获准在 `D:/projects/back/t17/chd-segmentation-assistant2` 使用最多2例真实病例，每阶段最多2次更新，最多200个验证文件、峰值8 GiB，结束清理副产物。验收报告保留在 `D:/projects/back/sync-reports/chd-segmentation-assistant2`，不超过10 MiB。
- 已获准将固定提交的 MedSAM2 官方源码保存在 `vendor/MedSAM2`、独立Python依赖保存在 `env/medsam2`，合计最多1 GiB、约1万个文件，保留复用。用户已指定使用官方权重及训练方法，MedSAM2_latest.pt保存在data/model_weights，容量156040129字节，保留供后续微调；每次训练核对官方SHA-256。
- 最终源码按用户要求放在 `D:/projects/project/chd-segmentation-assistant2`，继续保留助手2的独立工作目录和分支。
- 原始病例、个人路径配置、验证产物、环境和权重不提交。
