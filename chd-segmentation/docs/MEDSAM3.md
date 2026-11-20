# MedSAM3文本提示预测

本模块对应论文《MedSAM3: Delving into Segment Anything with Medical Concepts》，作者仓库Joey-S-Liu/MedSAM3，固定提交84f43118b5b67b50abad81d7c946681ebb97dcda。使用本地SAM3基座和医学LoRA v1，保持作者的1008图像变换、逐文本前向、阈值、NMS及二值掩膜缩放语义。原始作者源码作为外部依赖，未复制到本工程。

## 入口与当前验证范围

```text
python -B run.py models medsam3 inspect --config PRIVATE_CONFIG --case CASE_ID
python -B run.py models medsam3 predict --config PRIVATE_CONFIG --case CASE_ID --output OUTPUT
python -B run.py models medsam3 predict --config PRIVATE_CONFIG --case CASE_ID --output OUTPUT --stop-after-slices 1
python -B run.py models medsam3 evaluate --config PRIVATE_CONFIG --case CASE_ID --output OUTPUT
```

所有病例显式列出；多个病例重复--case。配置模板为configs/models/medsam3.yaml。该专用子入口不通过MedSAM2/Q训练分发，不继承旧模型的训练次数或豁免。没有MedSAM3训练入口。本次业务范围为文本提示预测及独立评价。

当前只进行了源码语法和参数帮助检查；官方权重、隔离环境、真实前向、恢复、空间回写、公共资源会话和评价的端到端运行尚未验收。没有用无权重、模拟模块或伪造结果代替验证。依赖下载、安装及病例运行另需明确产物许可。

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
