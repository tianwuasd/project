# Xu 2023 分割实验 · 第一阶段

本分支在冻结的来源队列上实现 Xu 等的四条三维分支。复用之前 CHD 工程按图 4 编写的骨架，改为背景和十个前景结构。输入缓存由原 0.3.0 工作流生成。论文分割主实验在 1,282 例开发集内三折验证，本工程固定 590 例训练、128 例开发。数字用于方法比较时须同时列出队列、预处理和标注差别。[论文，§4.2–4.3、§5.1、图 4、表 3、5、6](https://doi.org/10.1016/j.media.2023.102953)

## 在独立代码目录运行

进入本分支的 `dataset1282_analysis` 子目录。用预处理时已验证的研究解释器。将下面三个变量替换为本次任务的路径，输出放在预处理目录之外。旧项目的 updater 跟踪 main，分支部署使用独立 Git 检出。当前正在做的预处理可继续使用它自己的代码快照。

```bash
PY=/path/to/research/bin/python
D=/path/to/prepared/zmic-source-v030/hu-marginal
O=/path/to/private/xu-complete
"$PY" scripts/xu_seg.py inspect --prepared "$D"
```

只有 `complete.json` 及其病例划分、监督元数据、指纹拟合病例与训练合同一致的缓存才可用。研究臂采用十类均出现的 552 例训练、122 例开发选权重，标识为 `complete`。这项筛选只能验证类别出现，无法验证每个局部边界都已标全。预处理统计使用 590 例。要运行边际监督适配臂，将下面的 `ARM` 改为 `marginal`，同时换新的 `O`，**从两个定位器开始重跑**。适配臂使用全部 590 例训练、128 例开发。两个实验的差异同时包含病例纳入与监督策略。

```bash
(
  set -eu
  ARM=complete
  args=(--prepared "$D" --device cuda --profile paper --supervision "$ARM")
  for branch in crop64 crop128; do
    "$PY" -u scripts/xu_seg.py train "${args[@]}" --branch "$branch" --output "$O/$branch"
  done
  "$PY" -u scripts/xu_seg.py localize "${args[@]}" \
    --crop64 "$O/crop64/best.pt" --crop128 "$O/crop128/best.pt" --output "$O/roi"
  for branch in all64 all128; do
    "$PY" -u scripts/xu_seg.py train "${args[@]}" --branch "$branch" \
      --boxes "$O/roi/boxes.json" --output "$O/$branch"
    "$PY" -u scripts/xu_seg.py evaluate "${args[@]}" --branch "$branch" \
      --boxes "$O/roi/boxes.json" --checkpoint "$O/$branch/best.pt" --output "$O/eval-$branch"
  done
)
```

训练中断时仅重提被中断的那条 `train` 命令，保持相同路径并加 `--resume`。完整 epoch 原子写入 `last.pt`，按开发损失保存 `best.pt`。学习率、数据身份、模型实现、ROI 内容、监督臂和训练配置绑定检查点。上例使用单个获准 GPU，未实现多卡训练。正式 128³ 的 batch 4 需要在目标硬件上另做容量验收；出现显存不足时不要将缩水配置的结果当作论文规模运行。

## 几何、损失与评价

- 缓存空间尺寸随病例变化。定位器分别取整卷生成 64³、128³，再将预测前景的并集包围框映回缓存。并集取框是本工程对双路定位投票边界的明确选择。两路全空时报错，不使用真值框。每例是否退化为整卷单列于 `localization_qc.json`，正式模式下所有病例均退化会阻止发布 ROI 文件。ROI 文件及配套 QC 应放在一起。后续全结构分支在此框内各生成指定尺寸，使用线性强度插值、最近邻离散标签与统一体素中心约定。恢复坐标考虑原生 XYZ、网络轴置换、nnU-Net 裁剪及重采样。
- 3D 网络五层，64³ 起始 64 通道，128³ 起始 48 通道，空间关联分支与 U-Net 概率作逐点乘积。正式配置 batch 4、15 轮、初始学习率 2e−4，训练期后两段乘 0.1、0.01。末批使用实际剩余病例。全类病例用背景权重 1、前景权重 2 的 Dice，加普通交叉熵。缺类病例用既有边际监督。Adam、实例归一化的 affine 参数、激活斜率 0.01、奇数轮段落取整、旋转幅度 ±15°、缩放 0.9–1.1 及小幅弹性位移均是可审计的实现取值。完整网络计算得 90,461,879 / 50,957,351 个可训练参数，论文报告 84.9M / 51.2M，具体层级差异待作者实现核对。
- 每个分支可单独评价。输出分别汇总十类均出现的开发组及全部开发病例的已标结构。保存逐病例、逐类分母、均值和样本标准差。回到本例 Excel 毫米网格计算 Dice、HD95、ASD。选用六邻域表面体素中心、双向最近距离合并后取第 95 百分位及均值。未标类跳过；有真值却预测为空时 Dice 为零、距离未定义，单列空预测数与各指标分母；已确认真实缺如时另记预测体素数。十类都有可计算均值才给十类宏平均。加 `--save-predictions` 可导出带本地毫米网格的私有 NIfTI。

## 本地验收和待补材料

可用真实三例短测缓存运行下列命令，输出仅保存在私有目录。技术验证使用相同输入尺寸、缩小通道、batch 1、一轮训练；独立的 `--paper-forward` 用正式网络宽度验证真实 CT 前向。`--paper-step64` 额外检验正式 64³、物理 batch 4 下的单步训练容量。同一病例的四份增强样本仅用于容量验证。正式训练、论文性能和完整模型融合均须单独验收。

```bash
"$PY" scripts/xu_accept.py --prepared /path/to/three-case/hu-marginal \
  --output /path/to/private/new-acceptance --device cuda --paper-forward --paper-step64
```

本地检验中，两种正式宽度均完成真实 CT 前向，64³ 完成一次 FP32 反传和 Adam 更新。轻量定位模型的开发病例预测框覆盖整卷，尚无有效缩小 ROI 的证据。128³ 标签几何往返前景 Dice 约 0.964，衡量坐标和离散重采样。实际定位效果需用训练后的权重检查预测框体积、前景覆盖及逐结构遗漏。

起始段 `init64/init128` 需要可靠的 AO/PA 起始段监督。已核查的输入仅有常规十类掩膜，作者数字字典、起始段监督与血池边界生成规则仍待核实。图 4/§4.3 的七类语义与表 3 的 11 通道记录有差异。后续血池 2D U-Net/两层 BiConvLSTM、六路加权融合及 AO/PA 图优化独立验收。在补齐前仅报告有实测依据的三维分支结果。
