# 先心代码助手2接续

工作目录：`D:/projects/project/chd-segmentation-assistant2`。
工作分支：`codex/chd-segmentation-assistant2-medsam2-q-r2`。
基点：`298a75a50e18117bd62c741ebbc328b89dc8e7ad`。

本目录新增 `run.py models train/predict`、MedSAM2与Q路线一r2模型实现、配置、依赖清单和真实验收入口。原工作目录 `D:/projects/project/chd-segmentation` 保留给先心代码助手1，GPU分配、CPU供数和正式数据划分继续由其维护。后续整合时逐文件核对两个分支的差异，保留双方当前内容。

模型模块位于 `src/chd_segmentation/models/medsam2`、`q_route1`，共用执行位于 `models/workflow.py`。CLI接入仅在 `cli.py` 增加models分派；update.py增加Git工作目录支持和requirements-medsam2.txt同步白名单。

Q科学模块保留原网络、事件、求解、融合、定位目标与允许集合监督。统一入口保留M02/M07，正式训练使用A/B/C/D患者分组、分层候选和20%概率扰动、预热余弦及C组选模。关系R辅助头和独立临床参考未在本次输入中配置。

MedSAM2固定官方提交332f30d，使用官方Trainer、VOS采样/增强、模型、损失、优化器及检查点。初始化权重为官方MedSAM2_latest.pt，固定来源、容量和SHA-256在weights.py登记。序列为8帧、最多3目标、512分辨率，正式预算75epoch；提示式预测支持人工框或外部图像定位器提供的框，提示坐标为native_dhw。技术验收使用训练标签模拟提示并单独登记来源。

本地环境额外依赖与官方源码分别在env/medsam2、vendor/MedSAM2，均被Git忽略并保留复用。原始病例与验收报告在工作目录外；公开源码不包含病例文件或患者名单。

本机验证解释器为 `D:/projects/project/chd-segmentation/.venv/Scripts/python.exe`，基础环境只读复用。MedSAM2新增依赖仅在本目录env/medsam2，官方权重在data/model_weights/MedSAM2_latest.pt。移交到其他机器时需要按pyproject.toml、requirements-medsam2.txt和固定官方提交独立安装。入口为 `run.py models`；配置中的运行目录需要指向该任务的过程目录。

验收报告：`D:/projects/back/sync-reports/chd-segmentation-assistant2`，MedSAM2官方权重与训练器的两数据集记录位于medsam-official；单例完整检查点、并发保护及native_dhw预测回归记录位于medsam-atomic。真实验收程序每数据集使用一例病例、每阶段最多两次更新，保存后重载继续并导出原空间。Q使用96³视野和32³网络技术尺寸，MedSAM2使用32³视野及官方8帧512分辨率训练。报告限定于本地CPU技术功能；目标服务器CUDA/DDP、正式患者分组准入、长训练、心脏分割效果及临床终点仍需各自实际验收。

本分支按用户要求独立保留。后续合并与服务器发布使用明确的目标提交和单独的整合验收。

助手1main继承来源：公开提交6ef3fd4c51eb53da26faa54a147fc35ae15ebc40的chd-segmentation子树与本地助手1c28e191树摘要相同。助手2继承该项目全部变化，保留模型入口及工作目录支持。共享GPU资源调用位于models/execution.py，依赖training/resources.py；模型CPU缓存、官方数据worker、预取队列和读盘并发使用已准入配置。Q全局累积4样本按各rank份额归约；MedSAM2保留官方分布式Trainer和均等每卡batch。检查点保存逐rank状态并核对完整执行拓扑。

需要提交或上传的文件、服务器位置、Q分组及MedSAM2提示要求见 [服务器材料清单](ASSISTANT2_SERVER_FILES.md)。该文件同时说明当前发布边界：继承远端main与发布助手2模型是两个独立操作，服务器update.py以实际已发布main为准。
