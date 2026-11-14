# MedSAM2与Q的提交和服务器文件

## 本次代码

代码保存在D:/projects/project/chd-segmentation-assistant2，独立分支codex/chd-segmentation-assistant2-medsam2-q-r2。继承助手1公开main的6ef3fd4，对应本地项目提交c28e191。代码提交包含模型实现、共享资源调用、配置、测试和说明。

向GitHub发布时只提交上述源码；官方权重、依赖目录、病例清单、患者分组、预测和训练检查点分别管理。当前本地分支通过main继承并保留助手2模型修改，服务器update.py仍以已发布远端main为准；只有本分支内容正式发布后，服务器日常更新才会包含这些模型修改。

## 需要上传到服务器的材料

| 材料 | 本地位置或来源 | 服务器建议位置 | 状态 |
|---|---|---|---|
| 本分支完整代码 | 当前工作目录的受Git管理文件 | code/chd-segmentation | 本地维护，发布状态以交付提交为准 |
| MedSAM2官方源码 | vendor/MedSAM2，固定提交332f30d | code/chd-segmentation/vendor/MedSAM2 | 本地已有；服务器也可自行取得相同提交 |
| MedSAM2官方权重 | data/model_weights/MedSAM2_latest.pt，156040129字节 | data/model_weights/MedSAM2_latest.pt | 本地已有，SHA-256见weights.py |
| MedSAM2额外依赖 | requirements-medsam2.txt | 独立MedSAM2依赖目录，或已核对的训练环境 | 在服务器安装，按其Torch/CUDA构建核验 |
| Q基座官方权重 | 既有small_ep4k.model | data/pretrained/STU-Net/small_ep4k.model | 复用已审计的STU权重 |
| 两数据集审计清单及冻结划分 | 助手1既有私有server-inputs | course/chd-segmentation/inputs | 复用已有材料及bind-inputs入口 |
| Q患者分组文件 | A/B/C/D病例名单及患者身份 | course/chd-segmentation/inputs/q-groups.yaml | 正式训练前需提供或审核 |
| 两模型实际路径配置 | configs/models中的模板复制后填写 | course/chd-segmentation/configs/models | 需按服务器路径填写 |
| MedSAM2预测提示 | 人工框或图像定位器输出 | course/chd-segmentation/inputs/prompts.yaml | 提示式预测时需要；训练提示由训练标签产生 |

env/medsam2是本机额外Python包，服务器应按依赖清单安装并实际导入核验。既有Windows解释器和环境文件供本机使用。原始影像继续读取服务器既有data目录。

MedSAM2依赖清单中的TensorDict0.15.6要求PyTorch2.11及以上，本机已核对PyTorch2.12.1。服务器需使用相容的训练环境。vendor/MedSAM2需连同其自身Git元数据迁移，或在服务器独立检出固定官方提交，以便运行前核对源码身份。

将模型配置复制到course目录后，需要重新填写official_root、dependency_root、pretrained、run_dir和work_root。模板中的相对路径供本机工程目录使用，服务器配置应填写服务器上的实际路径。

Q的A训练上游、B训练局部、C用于选模、D独立评价，四组患者相互独立；预处理统计拟合限于A。当前尚未生成正式Q分组文件。组名相同不能代替患者身份核查。local阶段需要已完成的base和localizer检查点；base和localizer阶段的配置parents应为空。

## GPU及运行设置

两模型的execution字段复用助手1资源配置。work_root需与同一项目其他任务一致，使resources目录统一登记设备、CPU和RAM。支持1—4张卡、自动选择或gpu_ids手动选择；仅在全部可见卡繁忙且存在本模型、阶段、数据身份和尺寸的真实显存记录时评估共享。显存余量和等待窗口沿用main设置。

MedSAM2的batch是全局序列数，默认2，两卡各1；四卡时需显式设置batch4。官方DistributedSampler使用各卡相同batch，不能均分时启动前直接报错。Q的accumulation=4是r2一次优化器更新的全局样本数，1/2/3/4卡分别分担[4]/[2,2]/[2,1,1]/[1,1,1,1]；各卡累积的梯度按全局4样本求和归约，再统一裁剪和更新。

MedSAM2保留官方8帧512训练、75epoch日程和官方损失优化器。实际DataLoader进程数量来自共享CPU预算，每个worker的prefetch_factor受已准入队列上限约束，并登记实际值。Q在每个训练进程内同步读取和缓存病例，共享预算限制缓存及读盘并发。训练进程内部单线程；GPU计算由各获准设备并行执行。

完整恢复要求相同进程数量和批量份额；MedSAM2还要求相同数据worker数量，以保留随机抽样过程。可重新选择同数量物理显卡。Q每次完整更新、MedSAM2每个完整epoch保存检查点；压力中断后从最近完整边界恢复，未提交的计算重新执行。

从项目根目录执行，路径配置和准备描述需事先填写完成：

```bash
python -B run.py models train --config "$CFG/medsam2.yaml" --prepared "$PREP" --stage sequence --device cuda --gpus 2 --batch 2 --workers auto
python -B run.py models train --config "$CFG/q_base.yaml" --prepared "$Q_PREP" --stage base --device cuda --gpus 2
python -B run.py models train --config "$CFG/q_localizer.yaml" --prepared "$Q_PREP" --stage localizer --device cuda --gpus 2
python -B run.py models train --config "$CFG/q_route1.yaml" --prepared "$Q_PREP" --stage local --device cuda --gpus 2
```

用--gpu-ids 0,2替换--gpus 2可手动选卡。恢复同一任务追加--resume。模型预测在独立单进程内通过同一GPU准入，使用一张获准设备，保持协调进程不初始化CUDA。

CUDA/NCCL、MedSAM2多卡容量、正式输入尺寸及持续提速以目标环境实际验收为准。本次本地检查不启动正式训练。
