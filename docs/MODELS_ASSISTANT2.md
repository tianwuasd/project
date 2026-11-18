# MedSAM2 与 Q 模型

研究编号 Q-20261108-路线一，版本 q-route1-20261111-r2。助手2工作目录为D:/projects/project/chd-segmentation-assistant2，开发分支为codex/chd-assistant2-training-prediction-20261117，完成真实流程及审核后整合main。旧在线任务继续使用其冻结源码和运行目录。

## 文件职责

| 文件 | 职责 |
|---|---|
| `models_cli.py` | 独立模型训练与预测命令 |
| `models/workflow.py` | 读取准备描述、来源身份、优化器更新与完整保存/恢复 |
| `models/prediction.py` | 普通及标签框预测、测试子组选择、原网格导出和有提示评价 |
| `datasets/q_split.py` | 从1282冻结来源生成可迁移Q患者组合及配置 |
| `models/execution.py` | 调用助手1共享GPU/CPU准入、批量份额、进程登记和隔离预测 |
| `models/q_route1/` | r2局部网络、邻边事件、解析梯度、三步求解、定位器和一次概率融合 |
| `models/medsam2/official.py`、`worker.py` | 官方Trainer、有效配置、隔离运行与官方检查点恢复 |
| `models/medsam2/data.py`、`adapter.py` | 原始网格数据接口、官方VOS采样/增强及提示传播 |
| `models/medsam2/weights.py` | 官方发布权重、容量与SHA-256 |
| `configs/models/` | 每个模型的可修改配置 |
| `tests/verify_models_real.py` | 两套数据真实病例的有界验收及自动清理 |

## 输入与类别

公共离线合同继承助手1确定提交d68e11c，对应公开main 277c11a；脚本更新白名单继承7b50f5f/a0750b3。两个模型通过training/resources.py共用资源登记和准入策略，实际进程、批量份额和供数配置写入resources.json；完整恢复同时核对execution-topology.json。默认只使用空闲GPU，数量不足持续等待。显卡与材料说明见[服务器文件与GPU设置](ASSISTANT2_SERVER_FILES.md)。

先通过`prepare`生成来源描述，再使用`models preprocess`生成完整离线描述。训练和预测只读取兼容离线输入，影像和标签原件保持只读。Q内部类别顺序固定为背景、LV、RV、LA、RA、Myo、AO、PA、PV、SVC、IVC。CHD1282原始类别显式映射；ImageCHD输出回到其原始8类，PV/SVC/IVC保留未知监督，输出时其概率并入背景。

Q读取`stunet-ct-nnunet-bounded-v1`的float32影像、标签、前景位置和实际来源几何。新建Q预处理仍只在A拟合统计；直接消费用户明确指定的已有STU来源时，按数据集、输入形式和当前病例名单读取，保留该来源的真实计划与统计记录。训练阶段只要求该阶段训练及验证病例，预测只要求选中病例，已有数组不因当前实验分组变化自动重建。Q的定位目标、候选、上游冻结预测和局部求解仍按r2处理。

MedSAM2原生来源使用`medsam2-native-window-uint8-v1`。也可消费已有STU来源：auto按实际语义选择原生uint8直接读取，或用记录的mean/std撤销线性标准化，再按来源灰度窗转换。原先裁剪丢失的信息不能恢复。缺少语义时明确终止；normalized_percentile仅在用户显式选择时使用。官方八帧采样、增强和512缩放在训练时执行，网络、损失和优化器保持官方定义。当前实验名单独立于数组来源；CHD1282的Med实验保持其冻结训练和验证角色。

指定独立数组来源时，在模型配置使用`input_source: {path: /path/to/existing/preprocessed, intensity_mode: auto}`，`--prepared`仍提供当前实验的数据集和角色。path也可指完整离线描述文件，相对路径按配置文件目录解析。已有完整离线描述传给models preprocess时直接复用，生成小型引用描述，不重复读取权重、扫描原件或复制数组。

新预测病例先使用冻结训练计划和新清单运行prepare，再运行对应models preprocess。普通预测的离线数据可只包含图像；标签框评测需要单独声明带标签输入。

正式Q训练需要 `groups` 文件，以 A、B、C、D 为键、病例ID列表为值。基座和定位器训练A，局部网络训练B，C用于验证选模，D保留测试。声明患者单位时核对真实患者互斥，ImageCHD病例单位按case_id互斥并按病例评价；不补造患者身份。1282冻结组合保持患者单位及原角色，原训练患者按A:B=2:1拆分；A/B/C/D分别393/197/128/456例，操作见[1282组合与标签框](Q1282_AND_LABEL_BOX.md)。

## 命令

在本工作目录中运行，使用已安装项目依赖的解释器：

```text
python -B run.py models --help
python -B run.py prepare --config /path/to/prepare.yaml --output /path/to/source.json
python -B run.py models preprocess --config /path/to/model.yaml --prepare-config /path/to/prepare.yaml --prepared /path/to/source.json --root /data4/zhangruichen/data/chd-segmentation/preprocessed --output /path/to/prepared.json
python -B run.py models train --config /path/to/q-base.yaml --prepared /path/to/prepared.json --stage base
python -B run.py models train --config /path/to/q-localizer.yaml --prepared /path/to/prepared.json --stage localizer
python -B run.py models train --config /path/to/q-local.yaml --prepared /path/to/prepared.json --stage local
python -B run.py models train --config /path/to/medsam2.yaml --prepared /path/to/prepared.json --stage sequence
```

`--resume`读取同阶段完整检查点；`--stop-after`限制累计更新数，MedSAM2停止位置须位于完整官方epoch边界，轮内停止请求在执行更新前拒绝。MedSAM2正式预算为75个epoch。latest用于继续训练，best按独立验证指标选择；严格改善才替换best，损失相同保留较早状态。已有模型或检查点的目录需要resume或新目录；仅有候选、日志或资源记录的目录可正常初始化，同阶段操作锁和活动进程检查继续生效。

恢复核对科学设置、训练验证划分及监督语义。用户显式更换预处理目录、fold或spacing时继续允许复用模型；实际来源变化写入状态，选模比较从当前来源重新开始，Q候选抽样器按新输入构建，此时不声明输入不变的精确恢复。

提前停止默认关闭。Q默认patience=10次验证，STU及Med默认20个完整验证轮次，min_delta=0.0001；微小严格改善仍更新best。模型CLI提供--early-stop、--patience和--min-delta。Q预算以updates为单位，使用--epochs时必须同时提供--updates-per-epoch，显式updates与换算值必须一致。

MedSAM2保存官方 `checkpoints/checkpoint.pt`，同一文件包含模型、优化器、进度及`_chd`中的完整随机状态、提示随机状态和运行身份。写入完成并同步磁盘后原子替换，阶段锁在读取和写入前取得，持续到训练子进程退出。`official-config.json`记录实际完整配方。运行使用官方Trainer和train_epoch，数据增强、RandomUniformSampler、AdamW、余弦调度、0.9逐层学习率衰减、0.1梯度裁剪均取自官方FLARE25配置。默认学习率5e-5、图像编码器3e-5、weight decay 0.1；bias与LayerNorm权重按官方配置不衰减。默认8帧、最多3目标、512分辨率、batch2、75epoch，batch和worker可按资源修改。

MedSAM2可在模型配置中显式设置distributed_backend: gloo。默认CPU选择Gloo、CUDA选择NCCL；库未提供所选后端时直接报告，不自动更换。Windows单卡CUDA/Gloo需在获准环境先做真实通信检查，参数级支持不表示完整模型或显存容量已验证。CPU预算包含最多3目标的计算图；本机16GiB范围的官方前向已因内存不足停止，完整训练与恢复仍需适当资源的实际验证。

Q基座配置的updates填写40000，定位器填写5000，局部网络填写20000；正式累积次数为4。基座学习率预热1000次、其他阶段500次，之后采用r2余弦日程。局部前2000次只监督初始分割及q，后18000次加入三步求解。`method`支持M07及同监督M02；候选按真实错误、正确关键结构及其他候选分层，空层重新归一化，概率扰动比例为20%。`key_references`可登记候选与独立关键结构参考，没有该参考时不会将候选登记为已正确关键结构。候选抽样和扰动随机状态均保存恢复。

```text
python -B run.py models predict --config /path/to/q-local.yaml --prepared /path/to/prepared.json --checkpoint /path/to/q/local/best.pt --output /path/to/result/q
python -B run.py models predict --config /path/to/medsam2.yaml --prepared /path/to/prepared.json --checkpoint /path/to/medsam2/sequence/latest.pt --prompts /path/to/prompts.yaml --output /path/to/result/medsam2
```

普通预测数据加载器不读取标签数组。新病例使用包含新清单的准备描述，数据集类别语义需与检查点一致；几何恢复采用实际选定来源的轴序、spacing及ROI。显式描述按其storage.cases读取精确版本；显式目录只解析一次当前病例索引。Q根据CT定位器产生候选，模型快照固定方法与候选等科学参数；强度规则及实际输入来源写入任务，同一结果目录的接续调用保持相同推断设置。MedSAM2普通提示文件登记人工或图像定位器来源；自动提示登记定位器检查点摘要。显式`--prompt-mode label-box`从标签生成框并输出有提示评价，规则和命令见[1282组合与标签框](Q1282_AND_LABEL_BOX.md)。MedSAM2提示坐标使用原图所选范围的DHW网格，二维框以XYXY记录，frame指D轴切片，提示文件的space字段为 `native_dhw`。结构冲突按最高logit选择，背景logit固定为0。输出恢复到源数组尺寸与affine。

普通预测生成predictions.json，累计记录整个任务已完成的病例、实际来源、固定模型及相对文件名，可用共享evaluate入口计算已知类别指标；summary中的new/reused只统计本次调用。预测开始保存selected-model.pt；Q同时保存基础模型和定位器参数，随后从该组合继续，不随父模型路径后续写出而改变。已完成病例核对小型来源记录及文件存在后复用，不对全部输出做内容扫描。结果根目录使用单写锁，每例写出与完成登记之间保存pending-case.json；接续时补交已写完的文件，尚未写完的自有文件重新计算该例。标签框模式生成累计evaluation.json，保留有提示评价含义。

```yaml
case_id:
  - source: manual
    space: native_dhw
    frame: 10
    class_id: 1
    box_xyxy: [20, 30, 80, 90]
```

MedSAM2使用[固定官方提交332f30d](https://github.com/bowang-lab/MedSAM2/tree/332f30d420f1d1b08e2a79b3ae6a602458808383) 的Trainer、SAM2Train、官方VOS数据与变换、MultiStepMultiMasksAndIous 和 SAM2VideoPredictorNPZ。读取原始影像网格，CT窗默认来自训练计划的拟合上下界，也可明确填写gray_window。按官方CT处理方式执行窗内截断及0—255转换，后续执行官方RGB、增强和标准化。未标类别不生成目标，训练提示从已知标注产生；测试标签仅在显式标签框评测模式用于生成框及评价。CUDA后处理扩展显式关闭。

官方源码在vendor/MedSAM2，额外依赖在env/medsam2，官方权重在data/model_weights/MedSAM2_latest.pt，均被Git忽略并保留复用。权重来自[官方Hugging Face仓库固定版本](https://huggingface.co/wanglab/MedSAM2/tree/e4a6f35edd7e091619cbc0750f462f1574e23955)，容量156040129字节，SHA-256为 `c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060`。下载交付时独立核验；常规训练直接由官方初始化器严格加载参数，预测按实际模型结构与参数键加载。

常规运行记录实际配置的权重路径、文件容量和已导入源码位置，沿用已有交付记录；未重新核实的输入明确标记configured_pretrained及official_verified=false。直接权重预测的训练更新数标为未核实，不据参数键把任意同结构模型写成官方零更新状态。

模型GPU统计分别记录Torch allocator峰值和外部进程显存。外部指标缺失时保留None及原因，完整peak_vram_bytes为None，不据此写入共享或自动降卡容量记录；已保存的训练结果继续完成收尾。普通可观测路径保持原Torch峰值与外部占用合并口径。

额外依赖列于requirements-medsam2.txt。运行需要实际可用的依赖与API；必要导入或参数读取失败时直接报告。模型预处理不依赖初始化权重或严格Torch版本文字。独立安装不修改助手1的Python环境。

序列由官方RandomUniformSampler选择连续8帧及首帧可见目标；原始NIfTI接口提供同一图像和标签网格，读取错误直接终止。预测通过FrameSequence逐帧转换。CPU运行使用真实RSS统计适配官方训练日志，CUDA继续使用官方显存统计；训练计算调用同一官方train_epoch。训练子进程仅接收运行必需的环境变量。

## 验收与范围

验证限于已批准的1–2例真实病例、小区域和累计训练尝试；首次、恢复、失败及跳步均按统一台账计数，不自动补次数。Med保持官方8帧512、完整模型、损失及优化器。每轮报告记录实际来源、角色、训练与验证、恢复、预测复用、原空间、耗时、资源及清理；实际未通过的路径明确列出。

本轮共享验证工作区为D:/projects/back/t17/chd-review-20261118，助手2使用独占子目录；报告位于D:/projects/back/sync-reports/chd-review-20261118/assistant2。阶段3复用同轮模型与数组，消费结束后由唯一所有者核对路径和进程再清理。历史短测仍以各自报告的时间和范围为准。

正式权重训练、目标服务器CUDA容量、多卡和临床终点评价分别记录实际结果。本分支接入代码的验收范围以报告为准。
