# 部分标注预处理：STU-Net-S，96主队列＋13辅助病例

## 选择方法

| 方法 | 纳入 | 用途 |
|---|---|---|
| `stunet-s` | 96例完整标注主队列 | 保留当前完整监督预处理基线 |
| `stunet-s-partial` | 同96例主队列＋13例缺MYO辅助池 | 为已接入的部分监督训练准备数据 |

两种方法共享原生nnU-Net变换和V2资源调度，不复制一套算法。当前新输出协议为`imagechd-stunet-s-main96-aux13-ct-v2`，修正原生取块所需的四列坐标；旧v1已知生产者缓存可由训练入口只读适配。源灰度问题病例继续排除，不做自动纹理修复；其他缺标组合也不会自动放入辅助池。

```bash
python run.py prepare --list-methods
python smoke.py --stage prepare --method stunet-s-partial --output outputs/partial-smoke
python run.py prepare --method stunet-s-partial --fold all
```

短测默认输出一例完整标注与一例缺MYO病例，使用真实原尺寸；`--cases`或`--cohort`可显式覆盖。全部110对始终审计，各折仍用完整主训练子集拟合，不拿短测的两例估计数据分布。

只处理辅助池，或只处理主队列：

```bash
python run.py prepare --method stunet-s-partial --cohort auxiliary --fold 0 --output outputs/partial-aux-fold0
python run.py prepare --method stunet-s-partial --cohort auxiliary --fold 0 --output outputs/partial-aux-fold0 --resume
python run.py prepare --method stunet-s-partial --cohort main --fold 0 --output outputs/partial-main-fold0
```

`--cohort`只选择输出，不改变审计、主队列划分和拟合范围。原`stunet-s`没有辅助池，对它使用`--cohort auxiliary`会明确报错。显式`--cases`必须属于所选输出组。

## 角色与统计

先复现原97例完整标注的历史角色，再按既有质量排除获得96例；新方法绝不把109例重新洗牌。

- 主队列三折的train/val/test数量依次为53/10/33、55/10/31、54/10/32。
- 13例辅助病例在三折中均为`aux_train`，不会成为验证/测试真值。
- 每折CT强度统计、median spacing与各向异性决策只使用该折主队列`train`。辅助图像不参与拟合，即使当前只选择输出辅助池。
- 文件对独立性已检查；患者级身份或重复扫描关系仍未认证，正式评价前需进一步核验。

## 标签保持什么，新增什么

源影像和源标签不改。衍生标签仍保存0..7和ignore8，保留nnU-Net裁剪标记-1。额外源编号映射规则不变；ignore8不是第九个模型输出。

缺MYO病例的标签0仍为0，但明确表示“真实背景或未标MYO”。程序不会凭空定位心肌，不会自动填5，不会把其他六类标注覆盖成伪标签。重采样后允许原本就缺少的MYO继续缺少，原本存在的任意目标类消失仍然报错，不提交该病例完成项。

每例在manifest、任务JSON、输出pkl和completion记录中具有相同的`annotation`：

```json
{
  "schema": 1,
  "status": "partial",
  "annotated_class_ids": [1, 2, 3, 4, 6, 7],
  "unannotated_class_ids": [5],
  "background_semantics": "background_or_unannotated",
  "requires_partial_loss": true,
  "recommended_loss": "marginal_ce_dice",
  "expert_boundary_review_verified": false
}
```

`status=complete`只表示七类编号均存在，不代表已经专家逐层复核。部分标注的pkl另存`cohort_group=auxiliary`和三折`roles`。`foreground_sampling_class_ids`仅列出六个已标类别；MYO没有前景位置。原生nnU-Net的annotated-area坐标在此只能解释为“非ignore存储区域”，不能推断背景已完整标注，`annotated_area_semantics`会明确提示其背景歧义。

## 训练消费合同：预处理不自动启动训练

`manifests/training_contract.json`和每折的`training_contract.json`会标明：

- 当前为`preprocessing_only`，新合同标记`training_implemented=true`、`training_executed=false`，可由`run.py train`消费。旧v1合同的false是历史阶段状态，不代表当前训练入口不可用。
- 辅助数据需要部分监督，不能直接交给忽略病例元数据的完整监督trainer。
- 对辅助源标签0，后续边际交叉熵须合并`P(背景)+P(MYO)`，Dice也须对应适配；不是只关闭MYO的Dice项。
- 下游数据加载器须把每例annotation随patch传给损失函数，使用`aux_train`混合采样；此处未选择混合比例或正式训练预算。

各折`splits.json`包含独立的`train`、`val`、`test`、`aux_train`列表。主数据集JSON中的`numTraining=0`仍表示这是预处理产物，并非可以直接启动原生训练的完整数据集计划。

上述边际监督原则依据[Marginal loss原文3.2节](https://arxiv.org/html/2007.03868v1)，以及[COSST方法IV-B–IV-D](https://arxiv.org/html/2304.14030v3)。本工程已接入相应部分监督loss，未声称完整复现这些论文的训练或取得性能收益；实际入口和验证范围见[训练说明](training-usage.md)。

## 日志、恢复与资源

从`status.json`看当前阶段和数量；总日志、逐例变换与完整错误在`logs/`，每秒资源在`logs/resources.jsonl`。`summary.json`分别列主/辅总数、本次请求病例、新处理/复用数量及失败；纳入109例不表示短测已经输出109例。

输出结构沿用[预处理说明](preprocessing-usage.md)。新增的分流元数据属于方法兼容合同。当前代码更新后严格源码签名会与0.2.0旧运行不同，恢复会在写入前拒绝：保留旧结果和旧代码快照，用新输出目录启动，不删除旧结果、不跳过签名。相同新版本、相同输入和参数可通过`--resume`复用完成项；不完整项重新计算。

V2 0.7.2 CPU/RAM80%、至少4GiB预留、预计原生/目标体积加权准入、独立读取闸门、每秒内存监督及有界恢复均保持。`--workers`是可选上限，不是承诺实际并发；大重采样病例可能超出本机RAM，执行器会停止并保留小病例成果。完整109例×三折应在有足够RAM的服务器分阶段验收，不自动降分辨率来掩盖容量不足。
