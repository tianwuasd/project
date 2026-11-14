# 1282 测试与私有输入迁移

## 冻结角色

1282 保留训练、验证、内部测试和两个外部设备子组。`source_test` 映射到内部测试，`target_primary`、`target_secondary` 映射到外部测试。原 `cohort.json` 的患者、原角色、排除标记和重复扫描索引保留，不重新分配患者。

读取原候选名单时沿用其原角色。读取旧 HU 工作流派生名单时，同时执行 `included` 和 `index_scan`：已排除病例及非索引重复扫描不进入当前运行，其原记录保存在迁移包中。

当前私有迁移包保留原候选名单、派生名单与对应审核政策。两份名单的父摘要、政策摘要及原清单/Excel 版本已核对。按旧审核和索引规则实际选择：训练590、验证128、内部测试127、主要外部测试298、次要外部测试31。原候选数量与最终纳入数量分别记录，不把候选计数当作实际训练数量。

预处理统计仅拟合训练病例。验证用于训练期间的模型检查；内部及外部测试不参与统计拟合、优化器更新和模型选择。各测试组使用同一训练检查点、类别协议和冻结预处理计划。

## 两个测试入口

训练完成后，使用对应训练配置运行：

```bash
python -B run.py test --config "$INPUTS/chd1282_stunet_fold0.yaml" --set internal
python -B run.py test --config "$INPUTS/chd1282_stunet_fold0.yaml" --set external
```

外部入口默认处理两个设备子组，分别输出 `target_primary`、`target_secondary` 指标。可用 `--group target_primary` 或 `--group target_secondary` 选择其中一组；`--case 病例编号` 可以手动限制病例，选中的病例必须属于所选测试组。

结果保存在配置的 result_root 下，按数据集、模型、折、运行名、`tests/internal` 或 `tests/external` 组织。每例只保存一份最终掩膜，外部子组共享预测目录；各子组独立报告类别 Dice、有效病例数及均值。未知类别不计入相应指标，参考为空但出现预测的类别记为0，两者均为空时不参与该类别均值。

相同模型与输入重复执行时复用已完成预测和评价。输入、检查点或评价协议改变会要求新的结果位置。单病例选择的评价与全组评价分别命名，已有掩膜继续复用。日志和最近测试状态保存在 course 内。

## 一处迁移输入

私有迁移包集中在用户选择的目录，包含两数据集的审计清单、冻结划分、1282 原审核政策，以及1282固定fold0和ImageCHD三个fold的配置。包内不包含影像副本、训练检查点或环境。已有 STU-Net 权重继续从 data 中读取。

上传整个包到服务器的输入目录后，在项目根目录执行：

```bash
python -B run.py bind-inputs --directory "$INPUTS" --root "$ROOT"
```

`ROOT` 是包含 code、course、data、env、result 的服务器根目录。默认数据目录使用 data 内的 `dataset1282_CHD_CT_back` 和 `ImageCHD_dataset`；目录不同时用 `--chd-root`、`--image-root` 指定。已有权重可通过 `--pretrained` 指定。

绑定首先匹配审计记录的精确相对路径；需要时仅扫描文件名，使用原始文件名和审计输出名匹配。缺失、歧义或文件大小不同会停止，不复制或解压数据。原始 SHA256 在实际读取病例时检查。全部路径解析成功后写出清单和配置；相同绑定重复执行保持内容及时间不变。

迁移配置使用输入身份区分运行与计划目录，保护已有任务产物。绑定后可直接修改 batch、设备、训练轮数和其他配置；重新绑定会保留训练参数。

本地重新整理迁移包的工具为 `tests/build_transfer_inputs.py`，只读取既有 JSON/Excel 审计元数据；它需要可选依赖 `openpyxl`，服务器绑定与运行不需要读取 Excel。
