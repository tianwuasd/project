# 先心代码助手2接续

工作目录：`D:/projects/back/deliverables/chd-segmentation-assistant2`。
工作分支：`codex/chd-segmentation-assistant2-medsam2-q-r2`。
基点：`298a75a50e18117bd62c741ebbc328b89dc8e7ad`。

本目录新增 `run.py models train/predict`、MedSAM2与Q路线一r2模型实现、配置、依赖清单和真实验收入口。原工作目录 `D:/projects/project/chd-segmentation` 保留给先心代码助手1，GPU分配、CPU供数和正式数据划分继续由其维护。后续整合时逐文件核对两个分支的差异，保留双方当前内容。

模型模块位于 `src/chd_segmentation/models/medsam2`、`q_route1`，共用执行位于 `models/workflow.py`。CLI接入仅在 `cli.py` 增加models分派；update.py增加Git工作目录支持和requirements-medsam2.txt同步白名单。

Q科学模块保留原网络、事件、求解、融合、定位目标与允许集合监督。统一入口保留M02/M07，正式训练使用A/B/C/D患者分组、分层候选和20%概率扰动、预热余弦及C组选模。关系R辅助头和独立临床参考未在本次输入中配置。

MedSAM2固定官方提交332f30d，使用真实SAM2Train、官方损失与NPZ传播器。序列连续，帧按需转换；提示式预测支持人工框或外部图像定位器提供的框。外部定位器需要自身已验证的检查点与提示来源登记。本次技术验收使用训练标签模拟提示，官方结构随机初始化，尚未下载官方预训练权重。

本地环境额外依赖与官方源码分别在env/medsam2、vendor/MedSAM2，均被Git忽略并保留复用。原始病例与验收报告在工作目录外；公开源码不包含病例文件或患者名单。

验收报告：`D:/projects/back/sync-reports/chd-segmentation-assistant2`。真实验收程序每数据集使用一例病例、每阶段最多两次更新，保存后重载继续并导出原空间。Q使用96³视野和32³网络技术尺寸，MedSAM2使用32³视野及2帧512分辨率训练。报告限定于本地CPU技术功能；目标服务器CUDA/DDP、正式患者分组准入、长训练、心脏分割效果及临床终点仍需各自实际验收。

本分支按用户要求独立保留。后续合并与服务器发布使用明确的目标提交和单独的整合验收。
