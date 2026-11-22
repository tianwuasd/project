# 分类模块与执行条件

本文件描述已编写的生产接口。尚未运行模型实例、训练更新或权重预测；不得把源码存在或静态检查当作数值恢复/科研性能验收。预处理基础候选为`3aaae97397dd397edb8a2a907c4716e4e3163b75`，后续分类候选独立登记。

## 输入与确认监督

预处理回执阶段必须为`normalized_xyz`，目标XYZ为215×215×85，模型输入BCZXY为1×1×85×215×215。训练读取float16存储并转float32，推理读取float32；均不重复按HU量化。`prepared_inputs`可逐例显式指向既有预处理目录，默认使用`data_root/aaoca-classification/<case_id>`。只接受方法、源文件stat和输出stat均匹配的回执，不直接接收STU/Q数组。

`diagnoses`指定canonical CSV或XLSX/XLSM；多工作表还需`diagnosis_sheet`。列名如下：

`patient_key,aaoca_present,origin_left,high_risk,reviewer,reference,review_status,ct_matched`

只有`review_status=confirmed`、`ct_matched=true`或`1`，且复核者、参考来源完整时采用非空0/1。其他状态及未完成CT对应者保持未知。起源、风险只用于确认AAOCA阳性；正常者两项必须为空。候选文字、资料文件夹和ImageCHD身份不参与确认。现有诊断候选Excel的研究审核仍由医生完成；缺canonical复核列时直接停止，不推断列义或临床结论。

`export-confirmed-labels --output <private内新CSV>`只输出明确审核后可采用的值，其余留空，不修改原表或冻结清单。`check-training`按同一审核接口统计三任务；实际`train`只核对所选任务，每个任务的训练和验证分别需两类。全体患者及别名隔离，预测保留患者不进入训练。

## 配置字段

`classification_recipe=author_native_index`保留源索引。正式train必须提供`direction_review`：`approved=true`、`reviewer`、`reference`、`recipe`及3×3的`reference_direction_matrix_RAS`。逐例实际归一化方向矩阵必须与审核参考一致，LPS/RAS相反输入不会自动通过。该字段必须来自真实影像方向审核，不可为启动任务编造。

探索性predict在明确使用author_native_index时允许作者参照尚未确认，记录direction_status=unverified、原审核记录及逐例实际源方向、源/有效源/裁剪/目标/网络仿射和世界位移。缺少审核不会生成reviewer或approval，也不会重排数组或猜测参照。如果显式approved=true，则仍使用同一严格方向核对，缺来源或实际方向矛盾直接停止。该探索性输入的作者训练分布等价性未验证，不因此宣称科学复现或临床有效性；train及论文效果声明的要求保持。

顶层`coverage_reviews`按实际`case_id`索引；每项含与本次回执一致的`preprocessing_id`，并分别含`ostium`和`proximal_course`人工判读。两项各有`status`（`verified`、`unverified`、`not_evaluable`）和`included`。已核实时included须为布尔值；其余状态保持null。已核实或不可评价均需非空`reviewer,reference,review_date`，日期采用ISO年月日。来源须能追溯同一CT与审核范围，目标近端由阅片者明确，不能由LV/AO接触点或单个开口点推断整段包含。源码只验证登记完整性，不自动完成阅片或验证来源真实性。

缺少覆盖记录按`unverified`读取。正式train要求所选各例的开口和目标近端均独立`verified`且included=true；缺项、不可评价或截断在模型创建前停止。探索性predict记录两处真实覆盖状态，允许未核实、不可评价或明确截断，已有记录的身份及字段仍须一致。预测回执明确evaluation_scope为探索性且仅采用已确认参考计算指标，paper_performance_validated=false、clinical_diagnostic_validity=not_established，不因输出概率宣称论文标准覆盖或临床诊断有效。方向按上述探索性记录及显式批准时的一致性要求独立检查。

登记、原预处理、check-training及标签导出继续读取既有配置，不强制这些新增字段；check-training只检查监督，不代表全部正式运行条件已通过。现有未知标签与两例未核实覆盖保持原状态，迁移配置不能自动生成确认值。

`classification`中的通用字段：

|字段|要求|
|---|---|
|`task`|`aaoca_present`、`origin_left`、`high_risk`之一|
|`strategy`|正式训练显式选择下表中的实际初始化策略；作者预测策略在每个权重描述中声明|
|`device`|明确`cpu`或可用`cuda:N`，继承获准可见设备范围|
|`estimated_host_working_bytes`|可选正整数；省略时依据真实MONAI meta参数形状、输入和卷积工作数组估算，另计当前RSS|
|`estimated_gpu_working_bytes`|可选正整数；省略时自动估算，空闲显存扣除max(4GiB,10%总显存)后须容纳|
|`artifact_budget_bytes`|可选正整数覆盖；默认按实际参数与状态估算检查点，或按预测病例/成员数估算结果|

CPU线程由顶层`cpu_threads`控制，当前数据供给为workers=0、prefetch=0。主存采样和设备选择集中在resources；超限终止本进程，检查点原子保存保护上一份完整状态。此机制不保证拦截瞬时OOM，也不构成操作系统硬隔离。没有GPU独占调度、扩卡或分布式分类验收。

## 训练与恢复

|训练strategy|适用任务及实际约束|
|---|---|
|`local_from_scratch`|仅aaoca_present；无pretrained_detection，按现有MONAI Kaiming初始化|
|`author_detection_transfer`|仅origin_left/high_risk；pretrained_detection须为检测任务的author_state_dict|
|`local_detection_transfer`|仅origin_left/high_risk；pretrained_detection须为检测任务的local_complete_checkpoint|

其他模式直接拒绝，当前未增加子任务从头训练或检测微调。迁移仍需已知父检测训练患者来源且与当前验证组无交集；作者资料无法提供时保持缺项。父权重描述显式选择真实文件/成员，程序不按子成员索引推断父成员。策略、覆盖审核及原有语义设置进入运行身份；完整checkpoint的state保存实际strategy和coverage_reviews。恢复需要身份及策略一致，旧checkpoint缺字段时停止，不据文件目录补造其历史策略。

`train`还需：`member_index`（0–4）、`seed`、`learning_rate_recipe`、`epochs`（2–300）、`batch_size`和course内`run_directory`。主配方`paper_lr_1e-3=0.001`，发布默认对照`release_default_lr_1e-4=0.0001`。float32为当前明确精度，其他精度直接拒绝。作者requested MONAI1.4.0，本机actual1.5.2；必要计算路径源码对照一致，复用现有依赖，不以版本字符串阻止。实际网络构造核对阶段数、通道、分类形状和分组，权重严格加载。依据见`MONAI_COMPATIBILITY.md`，不宣称forward数值已验。

资源默认计划由真实MONAI的meta参数形状和卷积/池化几何估算，不使用虚假病例或执行forward。训练激活采用保守倍数，CuDNN工作区等仍需实际容量核查；计划明确`capacity_measured=false`和`requires_actual_capacity_measurement=true`。自动估算不能保证实际峰值，运行中保护保留。用户具体获准的测试/训练产物预算优先，源码默认预算不会扩大代理的运行授权。

检测从头初始化。起源/风险还需`pretrained_detection`权重描述，含真实检测训练患者来源，不能与当前验证患者相交。BCEWithLogitsLoss每批均值、AdamW权重衰减0.01、梯度范数限1；每轮loss为批均值平均。余弦Tmax=floor(0.75E)、eta_min=1e-6，每轮验证后一步。验证loss小于等于历史best时更新best并清零计数，严格变差100轮停止。

默认每10次更新保存完整latest，训练/验证切换和轮末同样保存；`checkpoint_interval_updates`可明确选择正整数，技术恢复核查用1。latest包含模型、AdamW、调度器、早停、最佳loss、epoch/阶段、排列及下批位置、累加loss、Python/NumPy/CPU及所选CUDA RNG。MONAI增强每样本由已保存NumPy状态派生种子，TorchIO使用已保存全局随机源。只管理所选CUDA设备的状态，不初始化其他可见GPU。当前未启用AMP，scaler明确为空。

`max_update_attempts`限制整个运行累计更新尝试数。每批供数及forward/backward/step前，将累计次数写入独立小文件`update-attempts.json`，flush/fsync后原子替换；失败或中断也消耗已预占次数。恢复必须同时具备身份一致、次数不小于latest内次数的登记文件，缺失或损坏直接停止；累计次数取该文件，模型/optimizer、消费位置及RNG仍取完整latest。最近检查点之后丢失的计算需要重新尝试并再次计数，不自动退还次数。完整检查点保持既定间隔，每次预占只写小登记文件。达到上限时先保存完整latest并停止，不自动追加训练或验证。新的具体授权可以明确提高该停止上限并以`--resume`接续；模型/数据/划分等语义设置须保持一致。有latest时必须显式resume，缺latest或身份不同直接停止。

best和latest均为完整同一时点状态。保存前按真实参数量估算临时容量，并核对批准预算。五个初始化采用同一训练/验证患者划分，每次运行需显式选成员；本入口不会自行连续启动五次训练。增强按固定作者六类MONAI与三类TorchIO分支实现，常量归一化或增强后尺寸异常直接停止。原文/代码随机种子未知，本地受控种子是可恢复适配，不声称重现作者随机轨迹。

## 权重和预测

`predict`还需course内`prediction_directory`、显式`prediction_case_ids`、`ensemble_mode`和`weights`。仅处理冻结预测角色中的所选病例，不自动展开整个队列。模式为`single_model`、2–4成员的`partial_ensemble`或5成员的`five_initializations`。

每个权重描述必须给出`path,task,labels,author_release,source,format,member_index,strategy`。格式为`author_state_dict`或`local_complete_checkpoint`。作者权重strategy须为`author_pretrained`，还需非空`experiment_name`，按实际包内配置与出处登记，不能拿README示例代替验证。本地权重strategy须为上表三项之一，并与实际checkpoint.state.strategy相同。集成成员的strategy及experiment_name须一致。source与实验名均为有依据的声明，代码不自动认证作者训练策略或逐成员父关系。

已核实的早期fc键仅在显式`legacy_fc_keys=true`时于内存映射，严格加载且不改原件。本地检查点还核对真实成员、种子和同一患者划分身份；重复路径或本地重复种子不能充当多个初始化。预测成员provenance完整保留真实checkpoint.initialization，包括实际父权重描述与来源；作者裸state_dict没有这项历史状态时不编造父初始化，保留其来源描述。

各成员先sigmoid，再概率平均，严格大于0.5为1。后两项报告为以AAOCA为条件的原始分类分数；无对应确认参考者不计入指标。报告明确患者/扫描分母、未知与不适用数、成员概率、预处理身份和权重来源，无两类时AUC和平均精确率不可估计。单成员结果不标为五成员评估。

模型实例计算、训练、实际预测及新增依赖/权重下载须分别按具体许可执行。r2两例预处理已完成，结果及清理记录见`PREPROCESSING_ACCEPTANCE_20261121.md`；该验收不扩大分类模型运行范围。
