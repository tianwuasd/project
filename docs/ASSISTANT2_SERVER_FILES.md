# MedSAM2与Q的提交和服务器文件

## 本次代码

代码保存在D:/projects/project/chd-segmentation-assistant2，独立分支codex/chd-segmentation-assistant2-medsam2-q-r2。公共供数、主机协调与资源策略继承助手1完成测试和发布的确定提交60d6e5f17c81f1fea78690021fbf403b1ee61f37，对应公开main 2269b26a4abc8a4952acacfab0611af4ce375d4f，来源子树cba6c019b1bc248262cdd5d70bd29be7aa6719ec。模型专属内存预算、官方训练与恢复检查保留。代码提交包含模型实现、共享资源调用、配置、测试和说明。

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
| Q1282私有输入包 | D:/projects/back/deliverables/chd-segmentation-assistant2-inputs | course/chd-segmentation/inputs/q1282 | 已生成393/197/128/456例组合、来源证据和三个阶段配置；迁移后bind-inputs |
| 两模型实际路径配置 | configs/models中的模板复制后填写 | course/chd-segmentation/configs/models | 需按服务器路径填写 |
| MedSAM2预测提示 | 人工框、图像定位器输出或显式label-box评测 | course/chd-segmentation/inputs/prompts.yaml | 标签框模式无需另交提示文件；需要测试标签；训练提示由训练标签产生 |

env/medsam2是本机额外Python包，服务器应按依赖清单安装并实际导入核验。既有Windows解释器和环境文件供本机使用。原始影像继续读取服务器既有data目录。

MedSAM2依赖清单中的TensorDict0.15.6要求PyTorch2.11及以上，本机已核对PyTorch2.12.1。服务器需使用相容的训练环境。vendor/MedSAM2需连同其自身Git元数据迁移，或在服务器独立检出固定官方提交，以便运行前核对源码身份。

将模型配置复制到course目录后，需要重新填写official_root、dependency_root、pretrained、run_dir和work_root。模板中的相对路径供本机工程目录使用，服务器配置应填写服务器上的实际路径。

Q的A训练上游、B训练局部、C用于选模、D独立评价，四组患者相互独立；预处理统计拟合限于A。1282私有输入包已保留原验证、内部测试和两个外部子组，仅将原训练患者按2:1分成A/B，具体命令见[1282组合与标签框](Q1282_AND_LABEL_BOX.md)。local阶段读取已完成的base和localizer的best.pt；base和localizer阶段parents为空。

MedSAM2沿用原CHD1282准备描述和原训练590、验证128、内部测试127及外部298/31的角色。Q使用独立的A组拟合计划；命令中的PREP与Q_PREP分别填写对应准备描述。选择Q准备描述会使通用train角色仅包含A组，需要按实际研究设计明确选择。

## GPU及运行设置

两模型的execution字段复用助手1资源配置。work_root需与同一项目其他任务一致，使resources目录统一登记设备、CPU和RAM。支持1—4张卡、自动选择或gpu_ids手动选择；默认share为false，仅使用空闲卡，数量不足持续等待。共享需明确设置execution.share为true或传入--share-gpu；--no-share-gpu可覆盖已有配置。繁忙卡需要本模型、阶段、数据身份、尺寸及本卡batch的真实显存记录，并满足连续观察窗口、负载与显存安全余量。等待期间整体取得所需设备，候选卡逐项原因写入日志。

默认execution.wait_seconds为null，持续等待；可用`--wait-seconds 300`设置本次时限，`--wait-seconds forever`持续等待。手动停止可终止等待。logging.heartbeat_seconds默认15秒，case_details默认true；主进程输出阶段心跳。资源等待日志保存在run_dir/admission/<阶段>/events.jsonl，执行日志及rank日志在run_dir/<阶段>；资源不足且尚未启动的任务可使用原目录重新尝试。Q记录每次完整更新，MedSAM2记录完整epoch，病例读取、标签框和预测结束分别记录。

MedSAM2的batch是全局序列数，默认2，两卡各1；四卡时需显式设置batch4。官方DistributedSampler使用各卡相同batch，不能均分时启动前直接报错。Q的accumulation=4是r2一次优化器更新的全局样本数，1/2/3/4卡分别分担[4]/[2,2]/[2,1,1]/[1,1,1,1]；各卡累积的梯度按全局4样本求和归约，再统一裁剪和更新。

MedSAM2保留官方8帧512训练、75epoch日程和官方损失优化器。实际DataLoader进程数量来自共享CPU预算，每个worker的prefetch_factor受已准入队列上限约束，并登记实际值。Q在每个训练进程内同步读取和缓存病例，共享预算限制缓存及读盘并发。训练进程内部单线程；GPU计算由各获准设备并行执行。

共享budget.py读取模型专属memory.py：Q计算原图读取、计划重采样、整例概率、融合及各rank保留的数组；MedSAM2计算原生网格病例、每个Dataset保留的一例、官方8帧队列和预测传播记忆。病例构建并发与已保留病例分别计入。MedSAM2原生病例缓存不使用通用LRU额度，实际LRU额度为零；CPU完整训练按此前batch1实测峰值设置每份额18GiB的模型预算，另计协调进程、病例、worker和队列，容量不足时等待。预算属于估算，正式网格和CUDA容量另行实测。

resources.json分别保存请求值、解析值和实际分配；rank日志登记实际预留内存、缓存、worker、预取和病例并发。MedSAM2每个rank另外写出official-config-rankN.json，保留各rank实际应用的官方配方。

两模型使用按消费顺序的有界供数：当前批次在全部rank完成消费后才释放下一批构建额度；病例构建同时受并发数和字节预算约束。模型加载、数据准备、传输、验证和保存结束后通过独立主机检查进入通信。官方采样、Q随机调用及全局batch语义保持原约定，具体事件与验收边界见[模型供数说明](MODEL_ORDERED_SUPPLY.md)。

完整恢复要求相同进程数量和批量份额；MedSAM2还要求相同数据worker数量，以保留随机抽样过程。恢复准入固定原拓扑，资源不足时等待，原worker额度不足时也等待。可重新选择同数量物理显卡。Q每次完整更新、MedSAM2每个完整epoch保存检查点；压力中断后从最近完整边界恢复，未提交的计算重新执行。两模型在本次执行期间保持训练进程和worker数量，启动时按各自预算选择供数配置；STU运行期间的动态供数和变卡验收范围见RESOURCE_SCHEDULING.md。

完整训练恢复保留原CPU或CUDA执行类型，以保存对应随机流；CPU与CUDA之间可迁移权重用于预测。模型及优化器状态核对数值、dtype和实际设备；AdamW的step计数按PyTorch自身规定保留。初始化异常结束且没有检查点、写入中的模型状态或训练摘要时，startup.json保存失败文件清单，核对后允许在同一阶段目录重新初始化。额外文件或已有文件变化会阻止重试；已有检查点继续要求resume。初始化重试保留各次请求配置和失败记录。

从项目根目录执行，路径配置和准备描述需事先填写完成：

```bash
python -B run.py models train --config "$CFG/medsam2.yaml" --prepared "$PREP" --stage sequence --device cuda --gpus 2 --batch 2 --workers auto
python -B run.py models train --config "$CFG/q_base.yaml" --prepared "$Q_PREP" --stage base --device cuda --gpus 2
python -B run.py models train --config "$CFG/q_localizer.yaml" --prepared "$Q_PREP" --stage localizer --device cuda --gpus 2
python -B run.py models train --config "$CFG/q_route1.yaml" --prepared "$Q_PREP" --stage local --device cuda --gpus 2
```

用--gpu-ids 0,2替换--gpus 2可手动选卡。恢复同一任务追加--resume。模型预测在独立单进程内通过同一GPU准入，使用一张获准设备，保持协调进程不初始化CUDA。

CUDA/NCCL、MedSAM2多卡容量、正式输入尺寸及持续提速以目标环境实际验收为准。本次本地检查不启动正式训练。
