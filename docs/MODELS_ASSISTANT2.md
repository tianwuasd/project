# MedSAM2 与 Q 模型

研究编号 Q-20261108-路线一，版本 q-route1-20261111-r2。工作目录由先心代码助手2维护，分支 `codex/chd-segmentation-assistant2-medsam2-q-r2`，起点298a75a。原工程目录与 GPU/CPU 调度分支由先心代码助手1维护。

## 文件职责

| 文件 | 职责 |
|---|---|
| `models_cli.py` | 独立模型训练与预测命令 |
| `models/workflow.py` | 读取准备描述、来源身份、优化器更新、完整保存/恢复及原网格导出 |
| `models/q_route1/` | r2局部网络、邻边事件、解析梯度、三步求解、定位器和一次概率融合 |
| `models/medsam2/` | 官方代码身份核验、序列数据、训练损失、提示及记忆传播 |
| `configs/models/` | 每个模型的可修改配置 |
| `tests/verify_models_real.py` | 两套数据真实病例的有界验收及自动清理 |

## 输入与类别

先通过原有 `prepare` 生成准备描述，模型命令读取同一记录与实际计划；影像和标签原件只读。Q内部类别顺序固定为背景、LV、RV、LA、RA、Myo、AO、PA、PV、SVC、IVC。CHD1282原始类别显式映射；ImageCHD输出回到其原始8类，PV/SVC/IVC保留未知监督，输出时其概率并入背景。

正式Q训练需要 `groups` 文件，以 A、B、C、D 为键、病例ID列表为值。预处理统计拟合病例限于A；基座和定位器训练A，局部网络训练B。各组患者身份需要独立。上游检查点保存训练病例和准备描述身份；局部阶段核对这两项。

## 命令

在本工作目录中运行，使用已安装项目依赖的解释器：

```text
python -B run.py models --help
python -B run.py models train --config /path/to/q-base.yaml --prepared /path/to/prepared.json --stage base
python -B run.py models train --config /path/to/q-localizer.yaml --prepared /path/to/prepared.json --stage localizer
python -B run.py models train --config /path/to/q-local.yaml --prepared /path/to/prepared.json --stage local
python -B run.py models train --config /path/to/medsam2.yaml --prepared /path/to/prepared.json --stage sequence
```

`--resume`读取同阶段完整检查点；`--stop-after`限制本次运行的累计更新数，配置中总预算保持原值。阶段目录分别保存 `latest.pt` 和 `summary.json`；正式验证按患者平均损失选出完整 `best.pt`，损失相同保留较早权重。输出目录已存在时新训练会拒绝覆盖。

Q基座配置的updates填写40000，定位器填写5000，局部网络填写20000；正式累积次数为4。基座学习率预热1000次、其他阶段500次，之后采用r2余弦日程。局部前2000次只监督初始分割及q，后18000次加入三步求解。`method`支持M07及同监督M02；候选按真实错误、正确关键结构及其他候选分层，空层重新归一化，概率扰动比例为20%。`key_references`可登记候选与独立关键结构参考，没有该参考时不会将候选登记为已正确关键结构。候选抽样和扰动随机状态均保存恢复。

```text
python -B run.py models predict --config /path/to/q-local.yaml --prepared /path/to/prepared.json --checkpoint /path/to/q/local/latest.pt --output /path/to/result/q
python -B run.py models predict --config /path/to/medsam2.yaml --prepared /path/to/prepared.json --checkpoint /path/to/medsam2/sequence/latest.pt --prompts /path/to/prompts.yaml --output /path/to/result/medsam2
```

预测数据加载器不读取标签数组。新病例使用包含新清单的准备描述，训练计划和数据集语义需与检查点一致。Q根据CT定位器产生候选。MedSAM2提供提示式预测，提示文件必须登记人工或图像定位器来源；自动提示需登记定位器检查点摘要。提示坐标使用准备后DHW网格，二维框以XYXY记录，frame指D轴切片。每个病例可提供多个结构提示，冲突按最高logit选择，背景logit固定为0。输出恢复到源数组尺寸与affine。

```yaml
case_id:
  - source: manual
    space: prepared_dhw
    frame: 10
    class_id: 1
    box_xyxy: [20, 30, 80, 90]
```

MedSAM2使用[固定官方提交332f30d](https://github.com/bowang-lab/MedSAM2/tree/332f30d420f1d1b08e2a79b3ae6a602458808383) 的 SAM2Train、BatchedVideoDatapoint、MultiStepMultiMasksAndIous 和 SAM2VideoPredictorNPZ。输入在准备后的网格按每例0.5%/99.5%灰度范围转换，使用官方512分辨率与RGB标准化；该配方单独记录，心脏任务效果需实验。未知类别不生成训练目标，训练提示从已知标注产生；测试标注只进入独立评价。CUDA后处理扩展显式关闭。

官方源码在vendor/MedSAM2，额外依赖在env/medsam2，两者独立保留并被Git忽略。额外依赖固定在requirements-medsam2.txt；本次解释器使用PyTorch2.12.1+cu126、torchvision0.27.1+cu126，TensorDict0.15.6需要PyTorch2.11及以上。部署时需要安装相容的实际构建。官方预训练权重保存在数据目录，配置明确提供路径；技术验证使用官方真实结构初始化。完整模型状态恢复后可以依据初始化摘要继续运行。安装不修改助手1的Python环境。

序列保持连续正向或反向切片顺序；两帧技术配置保留至少一帧用于传播训练。训练仅构造被选中的帧，预测通过FrameSequence逐帧转换，避免预先建立整例512分辨率RGB副本。灰度统计与主机安全余量在转换前核查。

## 验收与范围

验证限于用户批准的两个病例，Q使用96³技术视野、32³基座和局部训练输入，MedSAM2使用32³技术视野、2帧512序列。每阶段最多两次优化器更新，第一次保存后重载继续第二次。记录输入摘要、真实参数变化、阶段耗时、输出空间及原件摘要。技术提示由训练标签模拟，报告单独登记该来源。

测试工作区保存在 `D:/projects/back/t17/chd-segmentation-assistant2`，完成或异常退出时核验归属、记录文件数量和容量后自动删除；保留报告位于 `D:/projects/back/sync-reports/chd-segmentation-assistant2`。

正式权重训练、目标服务器CUDA容量、多卡和临床终点评价分别记录实际结果。本分支接入代码的验收范围以报告为准。
