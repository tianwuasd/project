# Xu 2023 固定划分解剖分割

进入本分支的 dataset1282_analysis 子目录，使用预处理时验证过的研究 Python。输入为已完成的 0.3.0 hu-marginal 目录，实验产物写在另一目录。入口脚本校验完成标志、训练统计、病例角色、标签及缓存几何；只读取此包。主训练组使用十类均出现的 552 训练 / 122 开发，边际损失组使用 590 / 128；缺标以 unknown 处理。每组独立训练、选权重，并报告全部开发例的可观察结构结果。训练病例的预测框来自见过这些患者的定位器，训练与开发覆盖率分开报告。正式队列的有效性需在固定开发集实测。本次范围为分割；流程没有诊断入口。

## 训练、取框与四路基线

以下命令使用 Bash，变量改为本次环境和私有目录。此分支要用独立代码检出；旧 updater 跟踪 main。定位和全结构模型均使用 3D U-Net，64³ 和 128³ 为两个分支的网络输入尺寸。正式模型分别有 90,461,879 / 50,957,351 个可训练参数，作者约 84.9M / 51.2M，细节差别尚未核实。正式配置为 15 轮、batch 4，初始学习率 2e-4，后两段乘 0.1 / 0.01；完整标注用加权 Dice 和普通交叉熵，部分标注另用边际损失。128³ 正式 batch 4 请先在目标硬件做容量验证。[Xu 等，图 4、表 3、§5.1](https://doi.org/10.1016/j.media.2023.102953)

```bash
(
  set -eu
  PY=/path/to/research/bin/python
  D=/path/to/completed/hu-marginal
  O=/path/to/private/xu-marginal
  ARM=marginal
  run() { "$PY" -u scripts/xu_seg.py "$@"; }
  base=(--prepared "$D")
  args=("${base[@]}" --profile paper --device cuda --supervision "$ARM")
  run inspect "${base[@]}"
  for b in crop64 crop128; do
    run train "${args[@]}" --branch "$b" --output "$O/$b"
  done
  run localize "${args[@]}" --crop64 "$O/crop64/best.pt" --crop128 "$O/crop128/best.pt" --padding-percent 10 --output "$O/votes"
  run audit-roi "${args[@]}" --candidates "$O/votes/candidates.json" --output "$O/audit"
  boxes="$O/audit/boxes.json"
  for b in all64 all128; do
    run train "${args[@]}" --branch "$b" --boxes "$boxes" --output "$O/$b"
    run evaluate "${args[@]}" --branch "$b" --boxes "$boxes" --checkpoint "$O/$b/best.pt" --output "$O/eval-$b"
  done
  run fuse "${args[@]}" --boxes "$boxes" --crop64 "$O/crop64/best.pt" --crop128 "$O/crop128/best.pt" \
    --all64 "$O/all64/best.pt" --all128 "$O/all128/best.pt" --evaluate --output "$O/fusion"
  run refine-vessels "${base[@]}" --fusion "$O/fusion" --evaluate --output "$O/vessels"
)
```

定位只读取 CT 和模型权重。两路硬分割先按体素中心对齐到 128³ 网格。同一位置两路均判为前景即计入定位目标，即使预测的结构名称不同。一票前景一票背景的位置记为未决。取一致前景的最小长方体，边缘向外取整映回缓存。论文 §4.2 给出多数投票取框，前景归并、平票和外扩规则由本项目明确指定。逐体素投票中的平票处理可参见 ITK。[Xu 原文](https://doi.org/10.1016/j.media.2023.102953) · [ITK 投票说明](https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1LabelVotingImageFilter.html)

**默认每条边的总长度增加 10%。** 例如 100 个体素扩到 110 个，两侧各加 5 个。新增体素数向上取整，奇数时高坐标侧多加一个，碰到图像边界时截到边界。三轴均可扩足且忽略取整时，框体积约增加 33.1%。用相同权重运行 `--padding-percent 0`，并指定独立的定位和审核目录，即可做严格投票的零边距对照。候选记录同时保存未扩框、历史框和前景并集框。

**逐例门槛。** `audit-roi` 读取已落盘的候选，再用预处理包保留的原网格参考标签检查。体积按外接边界统计，覆盖率按能从裁剪区实际还原的原图体素中心统计。空框、没有可还原的体素、整图、缓存或原数组网格上体积占比超过 95%、任何已标结构完全漏出，都会阻止发放整批训练框。训练组和开发组还各自要求每个已标结构的覆盖率第 5 百分位至少为 95%。两个阈值均是本项目的验收选择。缺标类单列分母。失败清单写入 `localization_qc.json`，只有全部通过后才生成 `boxes.json`。使用训练框的每个步骤都会复查摘要、来源权重、规则和每例实际范围。更换训练框或监督组后重新训练下游分支。

## 起始段、血池与融合

- build-targets 从预处理标签和已登记的解剖映射生成单独的私有目标目录。有疑问的标签编号应先复核。九个血池组成结构均有监督（出现或经核实缺如）才提供血池监督；其他病例仍可进入原全结构边际损失训练。三类血池目标为背景、血池和按物理距离生成的代理边界。外薄带默认参数为 1 mm，可提供经复核的内部界面。务必记录目标版本并盲审血管误粘连和真实瓣口/分流误截断。
- 起始段分支要求审核过的 AO/PA 起止端面、位于血管内的种子、审核编号和本地毫米坐标；无法确认的病例无该辅助监督。目标语义为背景、四腔、心肌及一个合并近端血管类。作者图 4 与表 3 在七/十一通道上存在差别；由端面派生的标签属于适配实验。两个 init 权重须成对参与六路融合。完整构造、复核字段以及尺寸见命令帮助与 xu_targets.py。
- 血池从标准化 3d_fullres 缓存按计划的轴序读取相邻七张同病例切片，保留物理宽高比并补边到 512²。先训练 2D U-Net，取倒数第二层特征训练两层双向 ConvLSTM。正式配置的可训练参数分别为 497,780,707 和 369,347；论文报告约 452.9M、0.2M。正式模式要求训练和开发都有合格监督。三例技术缓存缺少合格开发目标时，只有显式的 --technical-subset --profile smoke --technical-unvalidated 才允许做容量检查，输出 technical.pt，不能据此声称有效。
- fuse 的四路基线使用两个定位分支各一票、两个全结构分支各两票。加两个近端分支（各两票）后，合并近端类别的票只计入血管组，细类依赖全结构证据。可附加独立的血池权重约束区域生长，随后处理连通区域；每项适配分别评价。refine-vessels 调用中心线和半径加权图处理可判定的小血管碎片，保存逐例修改数、拒判原因以及优化前后的开发评价。Excel 局部毫米轴不足以判定患者真实头脚/左右，方向相关的作者路径与左右命名需额外方向依据。病例原生数组空间导出使用局部毫米仿射。
- 每个输出目录独立；支持恢复的训练/融合任务用同参数加 --resume。输出权重、模型和目标身份需匹配，检查点、派生标签和报告仅保存在私有输出目录。

以下是在上述 Bash 变量下使用辅助模块的命令（把它们放在同一 shell 中）。有审核记录时才传 --reviews，路径指定私有 JSON。起始段缺少合格训练/开发病例时，继续单独使用四路基线。

```bash
# CASE 为固定划分内的病例代号。在私有目录生成带毫米轴的预览和待核准模板。
run review-targets "${base[@]}" --case "$CASE" --output "$O/review-$CASE"
# 汇总人工核准的端面和种子后传入 --reviews。仅构建可用血池监督时可省略该选项。
run build-targets "${base[@]}" --output "$O/targets" --reviews "$O/reviewed-planes.json"
run train-blood "${base[@]}" --targets "$O/targets" --profile paper --device cuda --output "$O/blood"
for b in init64 init128; do
  run train-init "${args[@]}" --targets "$O/targets" --boxes "$boxes" --branch "$b" --output "$O/$b"
done
run fuse "${args[@]}" --boxes "$boxes" --crop64 "$O/crop64/best.pt" --crop128 "$O/crop128/best.pt" \
  --all64 "$O/all64/best.pt" --all128 "$O/all128/best.pt" --init64 "$O/init64/best.pt" \
  --init128 "$O/init128/best.pt" --blood "$O/blood/best.pt" --evaluate --output "$O/six-with-blood"
```

**短测与正式结果分开**。技术数据须显式 --technical-subset。轻量 --profile smoke 默认一轮；需要验证训练后取框时，在 train 命令增加 --pilot-epochs 和可选 --pilot-width 2、4 或 8；这些设置仅允许用于技术子集。脚本 scripts/xu_accept.py 单独检查定位与容量；退化框或覆盖不合格时返回失败。正式评价在原网格报告逐类 Dice、HD95、ASD、可评价分母及空预测，开发集同时承担选权重。
