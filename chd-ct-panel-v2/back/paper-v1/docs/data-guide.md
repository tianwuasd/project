# 数据规范

只支持已转换、脱敏的 3D NIfTI，不自动读取 DICOM。CT 数值应为 HU；不要提前做未经记录的窗宽映射。

## Manifest

CSV 列 `case_id,patient_id,split,image,label,initial_label`，路径相对 CSV 文件；`split` 为 train/val/test。患者同次或多次扫描必须全部位于同一集合。所有行的 image 与全心 label 必填，init 阶段额外要求 initial_label。

```csv
case_id,patient_id,split,image,label,initial_label
case001,subject001,train,images/case001.nii.gz,labels/case001.nii.gz,initial/case001.nii.gz
case002,subject002,val,images/case002.nii.gz,labels/case002.nii.gz,initial/case002.nii.gz
case003,subject003,test,images/case003.nii.gz,labels/case003.nii.gz,initial/case003.nii.gz
```

`validate-data` 检查病例/患者重复、缺文件、标签整数值、类别范围、图像标签 affine 与 shape。它无法识别被伪造 patient_id 的同一患者，或内容相同但路径不同的扫描；上游必须完成身份与重复数据治理。

## 内部类别

| ID | 全心 | 初始血管任务 |
|---|---|---|
|0|背景|背景|
|1|LV|LV|
|2|RV|RV|
|3|LA|LA|
|4|RA|RA|
|5|AO|MYO|
|6|PA|初始 AO/PA 合并|
|7|MYO|—|
|8|SVC|—|
|9|IVC|—|
|10|PV|—|

编号是本项目约定，不能假设外部数据使用同一编号。全心标注应保持论文的十个前景结构。不能把缺少的类别任意补成背景后声称训练的是完整任务。血池由除 MYO 外的九类组成，边界由三维六邻域腐蚀得到。

## 方向与距离

读取后只做轴翻转/排列为 RAS+；不重采样斜切影像。原始 affine 被保存，输出恢复原方向。图特征 EDT 假设体素轴互相正交：带 shear 的 affine 必须先由正规医学影像工具重采样到正交网格再提取距离特征。AO 相对 spine 的位置用 affine 的世界坐标 X 分量，正值表示偏右。最高/最低切片定位是近似；斜切体积的实际弓部需要额外精确定位。

训练时 all/init/blood 的 ROI 来自该训练样本标签；推理时 ROI 只来自 crop64/crop128 预测，不读取任何真值。真实全流程表现会受 ROI 预测偏差影响。

## 三折与独立测试

固定临床测试集独立于所有开发折。按患者生成三份 CSV，每份开发患者约为 train 56.7%、val 10%、test 33.3%，明确诊断与机器分层策略，稀有病种不能被简单随机划分遗漏。三折模型分别写入 `checkpoints/fold0` 等目录。

每一折的 test 都是开发集的 out-of-fold 部分，不是用于声称临床结果的独立 2468 例。阈值及图参数只能用开发集确定。训练命令不会读取 test 图像，但会校验清单路径存在。

本项目未自动混入本机其他心脏项目的数据。取得授权并明确映射后再在本地建立 manifest，所有医学影像、权重和逐病例结果保持在 Git 忽略目录。
