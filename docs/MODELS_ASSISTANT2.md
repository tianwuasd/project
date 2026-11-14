# MedSAM2 与 Q 模型

研究编号 Q-20261108-路线一，版本 q-route1-20261111-r2。工作目录由先心代码助手2维护，分支 `codex/chd-segmentation-assistant2-medsam2-q-r2`，起点298a75a。原工程目录与 GPU/CPU 调度分支由先心代码助手1维护。

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

本分支已继承助手1公开main的6ef3fd4项目内容，对应本地c28e191。两个模型通过training/resources.py共用资源登记和准入策略，实际进程、批量份额和供数配置写入resources.json；完整恢复同时核对execution-topology.json。显卡与材料说明见 [服务器文件与GPU设置](ASSISTANT2_SERVER_FILES.md)。

先通过原有 `prepare` 生成准备描述，模型命令读取同一记录与实际计划；影像和标签原件只读。Q内部类别顺序固定为背景、LV、RV、LA、RA、Myo、AO、PA、PV、SVC、IVC。CHD1282原始类别显式映射；ImageCHD输出回到其原始8类，PV/SVC/IVC保留未知监督，输出时其概率并入背景。

正式Q训练需要 `groups` 文件，以 A、B、C、D 为键、病例ID列表为值。预处理统计拟合病例限于A；基座和定位器训练A，局部网络训练B。各组患者身份需要独立。上游检查点保存训练病例和准备描述身份；局部阶段核对这两项。1282组合已按原训练患者A:B=2:1生成，A/B/C/D分别393/197/128/456例，操作见[1282组合与标签框](Q1282_AND_LABEL_BOX.md)。

## 命令

在本工作目录中运行，使用已安装项目依赖的解释器：

```text
python -B run.py models --help
python -B run.py models train --config /path/to/q-base.yaml --prepared /path/to/prepared.json --stage base
python -B run.py models train --config /path/to/q-localizer.yaml --prepared /path/to/prepared.json --stage localizer
python -B run.py models train --config /path/to/q-local.yaml --prepared /path/to/prepared.json --stage local
python -B run.py models train --config /path/to/medsam2.yaml --prepared /path/to/prepared.json --stage sequence
```

`--resume`读取同阶段完整检查点；`--stop-after`限制本次运行的累计更新数。MedSAM2停止位置需要处于完整epoch边界，其正式预算为75个epoch。阶段目录保存 `latest.pt` 和 `summary.json`；Q正式验证按患者平均损失选择完整 `best.pt`，损失相同保留较早权重。输出目录已存在时新训练会拒绝覆盖。

MedSAM2保存官方 `checkpoints/checkpoint.pt`，同一文件包含模型、优化器、进度及`_chd`中的完整随机状态、提示随机状态和运行身份。写入完成并同步磁盘后原子替换，阶段锁在读取和写入前取得，持续到训练子进程退出。`official-config.json`记录实际完整配方。运行使用官方Trainer和train_epoch，数据增强、RandomUniformSampler、AdamW、余弦调度、0.9逐层学习率衰减、0.1梯度裁剪均取自官方FLARE25配置。默认学习率5e-5、图像编码器3e-5、weight decay 0.1；bias与LayerNorm权重按官方配置不衰减。默认8帧、最多3目标、512分辨率、batch2、75epoch，batch和worker可按资源修改。

Q基座配置的updates填写40000，定位器填写5000，局部网络填写20000；正式累积次数为4。基座学习率预热1000次、其他阶段500次，之后采用r2余弦日程。局部前2000次只监督初始分割及q，后18000次加入三步求解。`method`支持M07及同监督M02；候选按真实错误、正确关键结构及其他候选分层，空层重新归一化，概率扰动比例为20%。`key_references`可登记候选与独立关键结构参考，没有该参考时不会将候选登记为已正确关键结构。候选抽样和扰动随机状态均保存恢复。

```text
python -B run.py models predict --config /path/to/q-local.yaml --prepared /path/to/prepared.json --checkpoint /path/to/q/local/latest.pt --output /path/to/result/q
python -B run.py models predict --config /path/to/medsam2.yaml --prepared /path/to/prepared.json --checkpoint /path/to/medsam2/sequence/latest.pt --prompts /path/to/prompts.yaml --output /path/to/result/medsam2
```

普通预测数据加载器不读取标签数组。新病例使用包含新清单的准备描述，训练计划和数据集语义需与检查点一致。Q根据CT定位器产生候选。MedSAM2普通提示文件登记人工或图像定位器来源；自动提示登记定位器检查点摘要。显式`--prompt-mode label-box`从标签生成框并输出有提示评价，规则和命令见[1282组合与标签框](Q1282_AND_LABEL_BOX.md)。MedSAM2提示坐标使用原图所选范围的DHW网格，二维框以XYXY记录，frame指D轴切片，提示文件的space字段为 `native_dhw`。结构冲突按最高logit选择，背景logit固定为0。输出恢复到源数组尺寸与affine。

```yaml
case_id:
  - source: manual
    space: native_dhw
    frame: 10
    class_id: 1
    box_xyxy: [20, 30, 80, 90]
```

MedSAM2使用[固定官方提交332f30d](https://github.com/bowang-lab/MedSAM2/tree/332f30d420f1d1b08e2a79b3ae6a602458808383) 的Trainer、SAM2Train、官方VOS数据与变换、MultiStepMultiMasksAndIous 和 SAM2VideoPredictorNPZ。读取原始影像网格，CT窗默认来自训练计划的拟合上下界，也可明确填写gray_window。按官方CT处理方式执行窗内截断及0—255转换，后续执行官方RGB、增强和标准化。未标类别不生成目标，训练提示从已知标注产生；测试标签仅在显式标签框评测模式用于生成框及评价。CUDA后处理扩展显式关闭。

官方源码在vendor/MedSAM2，额外依赖在env/medsam2，官方权重在data/model_weights/MedSAM2_latest.pt，均被Git忽略并保留复用。权重来自[官方Hugging Face仓库固定版本](https://huggingface.co/wanglab/MedSAM2/tree/e4a6f35edd7e091619cbc0750f462f1574e23955)，容量156040129字节，SHA-256为 `c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060`。训练前逐文件核对，官方初始化器严格加载全部参数。

额外依赖固定在requirements-medsam2.txt；本次解释器使用PyTorch2.12.1+cu126、torchvision0.27.1+cu126，TensorDict0.15.6需要PyTorch2.11及以上。部署时需要安装相容的实际构建。安装不修改助手1的Python环境。

序列由官方RandomUniformSampler选择连续8帧及首帧可见目标；原始NIfTI接口提供同一图像和标签网格，读取错误直接终止。预测通过FrameSequence逐帧转换。CPU运行使用真实RSS统计适配官方训练日志，CUDA继续使用官方显存统计；训练计算调用同一官方train_epoch。训练子进程仅接收运行必需的环境变量。

## 验收与范围

验证限于用户批准的两个病例，Q使用96³技术视野、32³基座和局部训练输入，MedSAM2使用32³技术视野、官方8帧512序列、batch1及worker0。每阶段最多两次优化器更新，第一次保存后重载继续第二次。记录输入摘要、真实参数变化、阶段耗时、输出空间及原件摘要。技术提示由训练标签模拟，报告单独登记该来源。

测试工作区保存在 `D:/projects/back/t17/chd-segmentation-assistant2`，完成或异常退出时核验归属、记录文件数量和容量后自动删除；保留报告位于 `D:/projects/back/sync-reports/chd-segmentation-assistant2`。

正式权重训练、目标服务器CUDA容量、多卡和临床终点评价分别记录实际结果。本分支接入代码的验收范围以报告为准。
