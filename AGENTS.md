# 先心代码助手2

本目录负责 MedSAM2 与 Q-20261108-路线一，版本 q-route1-20261111-r2。

- 工作分支为 `codex/chd-segmentation-assistant2-medsam2-q-r2`，基点为 `298a75a50e18117bd62c741ebbc328b89dc8e7ad`。
- 所有编辑、暂存与提交在本工作目录执行。原目录 `D:/projects/project/chd-segmentation` 由先心代码助手1维护。
- 保留本目录自己的 Git index；不切换原目录分支，不暂存原目录内容，不调用项目 update.py。用户已要求继承助手1已发布main：远端6ef3fd4c51eb53da26faa54a147fc35ae15ebc40的chd-segmentation子树与本地助手1提交c28e191逐文件相同。
- 共享GPU分配、资源登记和CPU预算沿用助手1的training/resources.py。MedSAM2和Q分别接入模型入口，保留各自训练方法、预算和数据分组。
- 已继承助手1本轮测试发布完成的60d6e5f17c81f1fea78690021fbf403b1ee61f37，公开main为2269b26a4abc8a4952acacfab0611af4ce375d4f，源码子树cba6c019b1bc248262cdd5d70bd29be7aa6719ec。公共STU供数、主机协调、字节额度和日志按该确定版本使用；模型预算、配置验证出口及设备恢复检查保持模型接口。默认只使用空闲GPU，数量不足持续等待，共享功能需明确启用。模型专属预算区分Q重采样/概率融合与MedSAM2原生序列；恢复保留原进程、批量份额及MedSAM2 worker数量。模型资源等待日志保存在run_dir/admission/<阶段>，执行日志保存在run_dir/<阶段>，未启动任务可在资源不足后重试。
- Q保留r2的类别、网络、损失、求解和一次整例融合。用户已要求基于原1282冻结划分生成Q组合：仅原训练患者按r2的A:B=2:1进一步分组，C保留原验证，D保留原内部及外部测试身份；统计仅在A拟合。
- MedSAM2使用官方序列训练及记忆传播。用户已授权标签自动生成框的正式评测；该入口明确登记标签访问及框生成规则。普通无标签预测继续只接受人工或图像定位器提示。
- 本次已获准在 `D:/projects/back/t17/chd-segmentation-assistant2` 使用最多2例真实病例，每阶段最多2次更新，最多200个验证文件、峰值8 GiB，结束清理副产物。验收报告保留在 `D:/projects/back/sync-reports/chd-segmentation-assistant2`，不超过10 MiB。
- 已获准将固定提交的 MedSAM2 官方源码保存在 `vendor/MedSAM2`、独立Python依赖保存在 `env/medsam2`，合计最多1 GiB、约1万个文件，保留复用。用户已指定使用官方权重及训练方法，MedSAM2_latest.pt保存在data/model_weights，容量156040129字节，保留供后续微调；每次训练核对官方SHA-256。
- 最终源码按用户要求放在 `D:/projects/project/chd-segmentation-assistant2`，继续保留助手2的独立工作目录和分支。
- Q与MedSAM2使用models/supply.py约束当前批次构建、公平样本顺序和病例字节预算；数据、验证及保存完成后通过CPU就绪检查进入设备通信。MedSAM2包装官方索引与数据集，保留官方训练主体；当前批次完成设备消费后才开放后续构建，日志及边界见docs/MODEL_ORDERED_SUPPLY.md。
- 原始病例、个人路径配置、验证产物、环境和权重不提交。
- 助手3的小样本完整流程保留独立工具目录、分支、配置、结果目录和更新入口；不将其功能自动合并到助手2，不修改其目录。按指定版本调用生产CLI或接口时，分别记录版本和产物边界。助手1向助手2同步公共修复继续按既定流程执行。
