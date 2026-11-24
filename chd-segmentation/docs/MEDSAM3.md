# MedSAM3医学LoRA微调与文本提示预测

本模块对应论文《MedSAM3: Delving into Segment Anything with Medical Concepts》，作者仓库Joey-S-Liu/MedSAM3，固定提交84f43118b5b67b50abad81d7c946681ebb97dcda。使用本地SAM3基座和医学LoRA v1，保持作者的1008图像变换、逐文本前向、阈值、NMS及二值掩膜缩放语义。原始作者源码作为外部依赖，未复制到本工程。

## 入口与当前验证范围

```text
python -B run.py models medsam3 inspect --config PRIVATE_CONFIG --case CASE_ID
python -B run.py models medsam3 predict --config PRIVATE_CONFIG --case CASE_ID --output OUTPUT
python -B run.py models medsam3 predict --config PRIVATE_CONFIG --case CASE_ID --output OUTPUT --stop-after-slices 1
python -B run.py models medsam3 evaluate --config PRIVATE_CONFIG --case CASE_ID --output OUTPUT
```

所有病例显式列出；多个病例重复--case。基础配置模板为configs/models/medsam3.yaml。医学LoRA微调另使用configs/models/medsam3_finetune.yaml，包含check-training、train和validate入口；专用子入口不通过MedSAM2/Q训练分发，不继承旧模型的训练次数或豁免。

当前通过了源码语法、六个命令的参数帮助、缺材料时的无模型/无目录写入检查，以及Windows独立Conda环境的真实依赖导入和原生接口检查。SAM3基座及医学LoRA文件已齐备；真实模型前向、训练更新、恢复、空间回写、公共资源会话和评价的端到端运行尚未验收。病例运行需按具体范围确认产物。

## 医学LoRA训练输入与恢复

微调从base_config配置中的医学LoRA继续，保留rank16、alpha32及完整原生SAM3，训练dropout取官方full_lora_config.yaml。原模型参数冻结，只有lora_A/lora_B参与更新。损失沿用固定作者train_sam3_lora_native.py中的BinaryHungarianMatcherV2、分类/存在性、框L1/GIoU、掩膜focal/Dice及一对多匹配配置；每次更新处理一张切片，可含多个已知类别查询。多类别查询共享该片图像编码。作者入口：[固定版本训练源码](https://github.com/Joey-S-Liu/MedSAM3/blob/84f43118b5b67b50abad81d7c946681ebb97dcda/train_sam3_lora_native.py)。

训练和验证病例名单显式指定，并保持冻结train/development角色及患者分离。直接读取已绑定原始清单，复用SourceReader的共享数组索引/作者间距核对，按基础配置coverage/slices选择切片。未知类别完全不生成训练查询；已知类别在某片缺如时保留该类别真实阴性查询，不要求每片阳性。语义标签按同类8邻域连通分量构造真实实例，框来自实例像素；不使用框或标签作为模型输入提示，文本来自固定类别映射。当前官方实例损失缺少像素忽略区域接口，遇到该区域明确终止，不将其作为背景监督。

已有预处理路径可通过--prepared/--preprocessed-source指定登记JSON：format为medsam3_training_slices_v1，samples每项包含case_id、slice、file和source_stamp；file指向已有NPZ，其中rgb为uint8[H,W,3]、labels为整数[H,W]、valid为bool[H,W]。source_stamp含image、label各自的[字节数,mtime_ns]，必须对应既有源清单。已有RGB不重复窗化，官方1008变换仅在构造模型输入时执行；输入仍需保留原始来源登记。

```text
python -B run.py models medsam3 check-training --config PRIVATE_TRAIN_CONFIG
python -B run.py models medsam3 train --config PRIVATE_TRAIN_CONFIG --stop-after 1 --deadline UNIX_SECONDS
python -B run.py models medsam3 train --config PRIVATE_TRAIN_CONFIG --resume --stop-after 1 --deadline UNIX_SECONDS
python -B run.py models medsam3 validate --config PRIVATE_TRAIN_CONFIG --checkpoint best --deadline UNIX_SECONDS
```

CPU按workers/prefetch有界预取真实切片；读取线程和Torch主线程合计受cpu_threads预算限制。训练必须显式选择CUDA，官方损失直接执行Triton内核。GPU按显式精度运行原生前向/反向，使用AdamW、warmup及cosine学习率。整机80%/项目100GB和显式ram_limit_bytes继续约束进程组；显存额度取配置上限与实际空闲扣除公共预留的较小值，至少保留1GiB。配置中的16GiB为上限，不代表实际分配或实测需求，不自动缩小模型或分辨率。

每次完整更新原子保存latest.pt；验证平均官方损失改善时保存best.pt。检查点含全部医学LoRA张量、优化器、调度器、随机状态、更新位置、最佳值和来源身份。暂停边界保存latest，释放模型及优化器后等待，再从检查点真实恢复；固定deadline不续时。另保存latest_lora_weights.pt、best_lora_weights.pt，直接把其路径写入已有预测配置的lora_weights即可使用训练结果，其他输入和预测命令保持一致。

训练检查点含优化器状态，容量大于仅推断LoRA；模板会保留best/latest及两份纯LoRA导出，实际测试前需按其总量申请产物范围。Triton、Torch、Hugging Face、Matplotlib和CUDA缓存及临时文件均定向到该训练目录的runtime-cache子目录，纳入同一产物范围。微调梯度、检查点恢复、CPU预取与GPU协同、训练后预测以及Linux训练均尚未实测，独立SAM3语义头的测试结论不用于本模块。

## 输入、空间和标签

显式data_root绑定已有CHD1282原迁移清单及冻结cohort；保留其患者角色，不把清单中的点号数据根解释成源码目录。预测仅从所选记录提取图像和空间/监督声明，不打开标签文件。评价为独立命令，此时才读取标签路径。

十类固定为LV、RV、LA、RA、AO、PA、Myo、SVC、IVC、PV，对应1至10，0为背景。英文提示在config.py固定，不依赖病例标签中的类别是否存在。未检出时该类无前景；多实例按检测分数进行掩膜融合；类别重叠时高分实例优先，同分时较小class_id优先。检测阈值和NMS阈值预先设置，不以测试标签选择。

RAS仅为经抽查支持的研究约定，原数组(i,j,k)的+i右、+j前、+k上。输入源记录必须有作者Excel逐例spacing及证据；图像原始头和affine完整记录，研究affine单独保存，内部原点为首体素中心零。图像/标签原点差不触发标签平移；评价使用显式shared_index_excel共享数组索引合同，不能据此宣称已恢复扫描仪物理坐标或已证明逐体素解剖对应。

每次按k取image[:,:,k]，转置为行j/列i，再灰度复制RGB，使用官方1008变换；返回时先恢复原切片大小，再转置回i/j。HU取NIfTI实际缩放值属于项目假设，证据通过intensity_evidence记录。固定CT窗默认[-160,240]为本工程设计参数；窗化线性映射到uint8，不采用逐片min/max。明确的windowed_uint8可以直接输入，标准化值必须提供mean/std；不尝试恢复已截断的信息。

输入压缩NIfTI复用models/stunet/streaming.py的有界解压，每个病例、每种原件只生成一份.nii映射及完成记录。恢复时核对已登记的来源路径、大小、mtime及空间合同，不重新解压；直接提供未压缩原件时原地映射。缓存不能跨不同来源合同误用，不重复扫描所有病例或重算原件摘要。原始数据只读。

## 覆盖、保存与恢复

coverage=all覆盖全部k；coverage=selected需要逐例指定零基切片索引，仅证明部分切片流程。每片完成后先原子提交slice-k.npy，再原子提交case.json的完成范围。新进程只计算未提交的切片；文件缺失、来源或科学设置改变直接失败。--stop-after-slices仅在完成规定新片数后正常停于提交边界，不代表请求范围已经全部完成。

同一worker只初始化一次基座和LoRA。每个切片的类别结果在内存融合，不保存PNG，不逐文本重载模型。全部请求范围完成后输出原数组形状的prediction.nii.gz，保留原图空间头；输出标签dtype改为uint8、强度缩放改为1/0。未计算切片填255，不能按背景解释。case.json记录覆盖和原始头内容、研究空间及变换。独立评价只读取已完成的请求切片，忽略未覆盖区域，并按清单处理未知类别、忽略标签；指标保留体素数及Dice。

切片事务的待提交文件在进程中断后可以重新计算；没有完成标志的解压文件只在同一已登记的专属目录重新生成。正式输出由调用方保留，微型验证必须由监督器先记实际耗时/速度/容量/峰值和退出证据，再自动清理本次独占产物，不删除原件或共享依赖。

## 环境、资源与权重

父进程复用主工程ResourceSession和SessionGuard；默认min(100GB,整机80%)、共享资源目录登记、准入等待、单worker及单片供数，不继承Q3局部测试豁免。GPU准入在公共显存预留之外增加显式estimated_vram_bytes预算，此值是保守估计，不能称为实测峰值；当前模板在8GiB显卡上可能持续等待，需要用户选择合法可行的资源与精度配置。不会静默切换CPU、精度或缩小分辨率。日志按阶段、解压块、病例、切片、模型加载和复用输出；worker有显式时间上限。

模型worker使用指定Python。启动前在该解释器的-I -B模式只读核对sys.executable、numpy版本/文件和torch文件；numpy须小于2，两项依赖均须来自该解释器prefix。spawn启动数据中的sys.path替换为该解释器实际路径及本工程源码路径，同时临时移除PYTHONHOME/PYTHONPATH并禁用user site；启动完成恢复父进程设置。该实现显式使用CPython multiprocessing.spawn准备接口，接口缺失即失败。子进程再次核对实际导入证据后才加载模型；独立环境未准备前，此路径隔离尚未经过真实子进程验收。

官方代码Git版本须固定且已跟踪源码干净；词表、配置、基座和LoRA须为本地有效文件。构建器设置load_from_HF=False，所有worker使用离线模式；基座detector权重严格加载，LoRA键集合和加载结果必须完整匹配。尺寸错误或缺失直接终止，不能换用普通SAM3或其他医学模型。

医学LoRA模型卡标记Apache-2.0；SAM3基座使用SAM License且需Hugging Face访问许可，不能将组合描述为单一Apache授权。上游作者仓库独立许可证仍需核实，未澄清部分保持外部依赖，不随本工程公开分发。依赖、词表、权重长期保留的位置和下载峰值需在执行前登记许可，本模块不自动安装或下载。

来源：

- https://arxiv.org/abs/2511.19046
- https://github.com/Joey-S-Liu/MedSAM3
- https://huggingface.co/lal-Joey/MedSAM3_v1
- https://huggingface.co/facebook/sam3
## 医学LoRA微调接入记录（2026-11-23）

目标：在现有独立分支，从现有MedSAM3-v1医学LoRA继续训练，复用原生SAM3完整图像模型、1008输入和官方检测/掩膜损失。保留现有预测入口及其他模型。

实施步骤：①增加独立微调配置与check-training/train/validate入口；输入直接引用冻结病例清单、作者间距和可选已有切片NPZ；②使用官方Datapoint/collator、匹配器及分类/框/掩膜损失，未知类别不生成查询，真实阴性保留；出现像素忽略区域则明确拒绝该样本，避免将其当背景；③CPU有界读取与预取，GPU BF16，复用公共准入及暂停检查；④保存LoRA、优化器、随机状态、步数、最佳验证值和来源身份，原子写入latest/best，另导出兼容现有预测的纯LoRA权重；⑤退出续训及资源暂停后的恢复必须读取真实检查点；⑥只做无产物检查，材料补齐和1–2例有限真实训练另按具体范围取得许可。

2026-11-24材料核对：医学LoRA与原生sam3.pt已齐备。原生权重大小3,450,062,241字节，SHA256为9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e；用户提供文件及长期保存副本均与facebook/sam3固定版本3c879f39826c281e95690f02c7821c4de09afae7的官方文件记录一致，未进行权重格式转换。该核对确认文件完整性，模型加载及真实训练尚未验证。

## Windows独立Conda环境（2026-11-24）

依赖清单为requirements-medsam3-windows.txt。已安装Python3.11.18、NumPy1.26.4、PyTorch2.9.1+cu126、torchvision0.24.1+cu126、triton-windows3.5.1.post24与Decord0.6.0，pip check通过。Triton采用维护者提供的Windows发行包，版本匹配依据为其[Windows说明](https://github.com/woct0rdho/triton-windows/tree/readme)。

tests/verify_medsam3_environment.py实际导入固定作者的infer_sam、lora_layers、模型构建器、Datapoint/collator、匹配器和官方损失，检查实际依赖属于独立解释器，并确认CUDA可识别。本机该检查耗时16.54秒，病例读取0、训练更新0；未创建模型，未执行CUDA内核，不据此认定显存容量、真实损失反向或模型恢复已通过。

```text
conda create -n chd-medsam3 --override-channels -c conda-forge python=3.11 pip "vc14_runtime>=14.42"
conda run -n chd-medsam3 python -m pip install --index-url https://pypi.org/simple -r requirements-medsam3-windows.txt
conda run -n chd-medsam3 python -B tests/verify_medsam3_environment.py --official-root OFFICIAL_MEDSAM3_ROOT
```

私有运行配置中的python应指向该独立环境的解释器；共享研究环境和显卡驱动保持原配置。安装日志、完整版本清单和环境检查记录保存于私有安装目录；完成后清理本次专用安装缓存，保留可复用环境与记录。

## 两例局部联测（2026-11-24）

tests/verify_medsam3_local_workflow.py使用显式指定的冻结train/development各一例，按各自体积深度的中间索引取一片，仅检查主动脉类别。原图与标签的SHA256、来源路径、患者分离、作者间距及共享数组索引检查通过；原始affine差异保留。真实窗化RGB与NPZ保存后重读逐元素一致，作者Resize/Normalize、实例构造和collator实际生成[1,3,1008,1008]图像及对应文本、框和掩膜；workers1/prefetch2读取检查通过。最后一次CPU输入与训练材料检查总耗时17.17秒。

完整联测状态为模型执行前受阻。当前主机80%安全内存余量不足声明工作集；共享GPU缺少此卡同配方的实测显存峰值记录，公共资源模块拒绝启动模型。训练更新0，恢复、验证损失和训练后预测均未执行，模型内存/显存峰值未测量。没有生成或填入虚假的共享容量记录。GPU容量身份按MedSAM3自身的权重/配置身份、查询数、阶段、精度、实现及依赖版本区分。

现有加速包括已预处理切片复用、有界CPU预取、线程预算、GPU BF16和作者训练激活检查点。训练的同片多类别查询共用图像编码；预测仍按类别执行完整前向，尚未实现跨类别图像特征缓存或跨片批量预测。实际吞吐提升尚无同条件对照结果。输入身份、未知类处理、非有限损失/梯度检查、原子检查点、优化器/RNG恢复、资源等待和固定截止构成运行保护；模型相关保护仍需真实训练进一步验收。

局部测试保留日志、配置及报告，自动删除切片副本和本次缓存；原始病例、共享依赖及正式权重保持原样。可用--prepare-only单独运行真实输入构造检查，该模式的通过只覆盖预处理。
