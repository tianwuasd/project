# 1282的Q组合与MedSAM2标签框评测

研究编号A-A → Q → Q-20261108-路线一，版本q-route1-20261111-r2，调整日期2026-11-14。

## Q组合

1282 Scheme A原训练患者按设备和年龄层分层，再使用固定种子20261104及患者摘要排序，按2:1细分A/B。全局人数采用最大余数分配。原验证成为C；原内部测试和两个外部子组共同构成D，测试子组身份分别保留。排除记录、索引扫描和原始患者身份沿用冻结来源。

正式组合为A 393、B 197、C 128、D 456例，每例对应一名患者。D包含source_test 127、target_primary 298、target_secondary 31例。A训练基座与定位器，B训练局部模型，C选择检查点，D评价。预处理清单将A登记为train、B登记为predict并保留标签、C为validation、D为test，因此自动统计仅拟合A。B局部训练通过显式Q分组读取。

生成独立私有包，再绑定机器路径：

```bash
python -B run.py models split-q1282 --source /path/to/original/server-inputs --output /path/to/q-inputs
python -B run.py bind-inputs --directory /path/to/q-inputs --root /data4/zhangruichen
python -B run.py prepare --config /path/to/q-inputs/q1282_prepare.yaml --output /data4/zhangruichen/course/chd-segmentation/prepared/q1282.json
```

数据或权重位置有差异时可在bind-inputs提供`--chd-root`、`--pretrained`。绑定会填写prepare及三个模型配置的机器路径，已有训练预算等手动配置保留。三个阶段依次使用q1282_base.yaml、q1282_localizer.yaml、q1282_local.yaml；局部阶段读取上游best.pt。

原STU-Net完整训练集的计划与检查点属于独立基线。Q需要自己的A组拟合计划和A组上游权重。Q正式训练逐项核对外部A/B/C/D完整名单与准备描述中的冻结组别、来源角色、重复或遗漏病例、患者互斥、拟合病例及上游训练名单。

本机私有包位于D:/projects/back/deliverables/chd-segmentation-assistant2-inputs，含冻结来源证据、可迁移影像清单、q1282_split.json、q_groups.json、配置及q_split_audit.json。影像原件保持原位置。患者名单和清单只按私有材料迁移。

## MedSAM2标签框

训练继续使用固定官方源码、官方发布权重和官方Trainer配方。显式启用标签框评测：

```bash
python -B run.py models predict --config /path/to/medsam2.yaml --prepared /path/to/prepared.json --checkpoint /path/to/sequence/latest.pt --prompt-mode label-box --test-set internal --output /path/to/result/medsam2/internal
python -B run.py models predict --config /path/to/medsam2.yaml --prepared /path/to/prepared.json --checkpoint /path/to/MedSAM2_latest.pt --official-weights --prompt-mode label-box --test-set external --test-group target_primary --output /path/to/result/medsam2/external-primary
```

`--official-weights`核验官方发布文件的容量及SHA-256，报告登记零次微调更新。省略该选项时核对本项目训练检查点的模型、范围、代码、配置及训练拟合计划身份。`--case`可限制病例；`--label-classes 1,2`可限定已知结构，默认提示全部已知且存在的前景结构。

每个结构选择标签面积最大的切片，面积相同取较早切片，生成该层二维包围框；使用病例和类别独立的确定性随机数，向四边分别外扩0—20个原始网格像素。`--box-seed`默认2024，`--box-max-shift`默认20，可设置0关闭外扩。框采用XYXY、右上界不包含的坐标约定。最大面积选层规则是本项目对多结构心脏标签的适配；外扩范围参考固定官方CT推断文件的标签框辅助函数。

每个结构分别初始化提示，正向和反向传播前重置记忆并重新加入同一框，沿完整所选体积传播。模型接收图像、结构编号、提示层和框；完整标签用于框生成及后续指标。结构冲突按logit最大值选择，背景logit为0。

summary.json记录每例标签摘要、提示层、紧框、外扩量、最终框、随机种子、类别存在情况及传播方式；evaluation.json仅评价已知、实际有提示的结构。未提示的缺如结构不计入该模式的Dice。结果属于参考标签框提示下的分割评价，自动定位能力需要独立评价。普通external模式使用人工或图像定位器提示，不读取标签数组。

## 有界验证

tests/verify_q1282_bundle.py核验冻结元数据、原训练覆盖、患者互斥、A拟合角色、D子组、机器绑定的可重复性及手动配置保留，不读取CT体素。

tests/verify_label_box_real.py使用ImageCHD ct_1003的32³技术范围、最多两个真实结构、官方发布权重及零训练更新，优先选择提示层大于零的结构，核验实际反向传播、框确定性、未知类别保护、普通模式提示要求、原网格导出、标签摘要和有提示Dice。summary记录每个结构、方向的实际传播帧数。该范围只证明接口功能。

tests/verify_model_guards_real.py分别验证错误Q分组被拒绝，以及同一真实病例在CPU预算不足时，Q和MedSAM2各连续两次准入失败后仍未创建执行阶段目录；不执行训练更新。资源等待日志独立保存在admission目录。
