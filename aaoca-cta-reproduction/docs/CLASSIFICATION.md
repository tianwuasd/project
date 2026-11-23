# 分类模块与执行条件

本文件描述生产接口及已测范围。2026-11-22已用一例AAOCA和一例ImageCHD完成预处理，严格加载作者三任务各model_1，并完成6次CPU预测；各任务均为单成员、未知参考2例、已确认参考0例。另已完成零更新检查点的同进程磁盘往返及CPU随机状态恢复。训练更新、完整训练续跑、CUDA、五成员集成和科研性能仍未验收。预处理基础候选为`3aaae97397dd397edb8a2a907c4716e4e3163b75`，分类生产修复为`8cd4ffc`，预测测试入口为`ea4b39c`。

## 输入与确认监督

预处理回执阶段必须为`normalized_xyz`，目标XYZ为215×215×85，模型输入BCZXY为1×1×85×215×215。训练读取float16存储并转float32，推理读取float32；均不重复按HU量化。`prepared_inputs`可逐例显式指向既有预处理目录，默认使用`data_root/aaoca-classification/<case_id>`。只接受方法、源文件stat和输出stat均匹配的回执，不直接接收STU/Q数组。

先心助理继续复用已有中文整理成果和逐条证据，交付含英文`data`、`evidence`、`definitions`的工作簿；可以另含完整诊断页。生产直接读取`data`，不另行生成CSV或要求用户重建表格。其他多表格式仍需`diagnosis_sheet`明确指定。

`diagnoses`指定实际CSV或XLSX/XLSM路径，原canonical必需列继续兼容：

`patient_key,aaoca_present,origin_left,high_risk,reviewer,reference,review_status,ct_matched`

新研究表增加`source_type,aaoca_certainty,origin_certainty,risk_certainty`，保留`ct_match_status`原状态。三个任务值为0/1/空白或unknown；逐任务确定程度为explicit/uncertain/unknown/conflict。`reference`为包含来源文书、页码、提取方式和日期等依据的非空JSON对象数组。多支或冲突保留coronary_artery=multiple/conflict及coronary_vessels，不强行填单侧origin_left。origin_site、course_type、coronary_findings及具体特征列均原样保留于source_fields，不从无狭窄、未提及或ImageCHD归属生成正常标签。

顶层`label_policy`默认`confirmed`，继续要求review_status=confirmed、真实reviewer/reference和CT匹配。用户允许的研究策略显式设`document_explicit`，采用source_type=document_explicit、对应任务certainty=explicit且有依据的0/1；不要求将原review_status伪改confirmed，也不填造reviewer。文书明确诊断/明确排除可用于研究标签，可疑、冲突、缺资料或未提及保持未知。起源/风险仅适用于该来源明确AAOCA阳性；风险未知只排除风险任务。

`research_sources`组合明确文书研究标签与逐病例`label_attestation`中的用户阴性确认。ImageCHD仅在实际用户已确认当前登记全体为AAOCA阴性并授权患者划分时使用；确认须包含来源、日期、原话、病例标识及当前影像大小。读取源登记时核对已确认登记的数量和stat，迁移绑定后核对每例大小并保留确认依据。该来源记录为`user_attestation`、review_status=user_attested，reviewer保持空白，confirmed_labels保持未知；不能伪造成医生确认。ImageCHD监督角色在这项显式确认存在时开放，默认预测角色继续兼容。起源及风险对阴性病例不适用，不把阴性编码成右冠或低风险。

该研究策略用于检测训练/验证取样、check-training、预测参考及指标，来源进入训练与恢复身份、完整检查点和逐例预测依据。未经用户确认的ImageCHD仍保持未知。检测类别与数据来源完全相关时，评估须明确此组成，不能将所得分数直接作为跨来源临床性能。冠脉像素分割不是当前分类入口的输入要求；已有心腔及大血管分割用于定位裁剪，影像覆盖仍须独立审核。

原复核状态及CT匹配字段原样保留。CT尚未匹配的确定文书标签保存于research_labels并单独计数，不能当作当前影像监督或性能真值。训练只核对所选任务在训练/验证中各有CT匹配的0/1，不因其他任务未知而停止该任务。研究标签策略、来源、确定性及依据进入新研究运行身份和完整best/latest，恢复时须一致；原confirmed默认身份计算保持。

新研究表及任何已有`matched_ct_images`字段的表，`ct_matched=true`还须有`matched_ct_images`（AAOCA数据根内具体影像相对路径的JSON数组）及`ct_match_reference`（实际文书/页码、检查日期或配对依据）。程序与当前manifest选用的完整相对路径比较，不用患者编号、文件名或任意历史CT日期替代体积身份。原始ct_matched声明保留于source_fields；缺路径/依据、当前体积不符或尚未选正式体积时，实际ct_matched为false并报告原因，不进入监督/真值。全登记审查分别统计available_ct_labeled与当前所选CT匹配数量，不自动重写冻结清单。研究运行身份和输出参考记录包含具体选用体积及配对依据。

正式表提供`aaoca_ct_matched/origin_ct_matched/risk_ct_matched`及各自`*_matched_ct_images`时，必须分别采用；ct_match_reference为含aaoca、origin、risk键的JSON对象，各值为配对证据数组。公共ct_matched/matched_ct_images仅代表各已填任务的共同CT，不替代逐任务配对。标签筛选、训练、预测真值/指标及研究运行身份采用所选任务的实际体积绑定；起源/风险的AAOCA阳性适用性依据该来源的原始明确诊断，不受检测任务另一CT配对状态覆盖。

原8列confirmed合同继续兼容（至多另有空白source_type列）：真实reviewer/reference、review_status=confirmed、ct_matched=true/1表示人工已确认当前manifest影像，读取时标注legacy_confirmed_attestation，无需重建旧表。该分支不接受新研究列或具体体积列，也不接受document_explicit/unknown来源；未选当前体积时仍不放行。导出保留原来源及逐任务CT字段，旧8列维持原合同格式，不把新研究标签升级为人工确认。

`export-confirmed-labels --output <private内新CSV>`仍只导出confirmed监督，不把研究标签升级。`check-training`按任务报告所选策略的匹配计数、确定研究标签、冲突、未知、CT缺项和缺少的类别。`check-label-registry`只读关联全体已登记CT患者及别名，按原冻结角色和outside_existing_freeze报告分布，不改原清单或自动分组。完整患者划分须依据真实新表分布另行审核，原12/4/4不是新数据的纳入上限。

预测报告保存所选label_policy、参考来源和逐任务适用性；研究来源指标与confirmed_reference_metrics分别报告，未知或CT未匹配者不进真值分母，不宣称论文效果或临床诊断性能。

2026-11-23已实际读取正式工作簿data页，并完成两例正式记录零文件核对及全登记只读汇总，4.8626475秒通过。正式工作簿/冻结清单stat未变；0模型/训练/GPU，内存峰未测量。实际私有类别数量和配对缺项留存私有交付说明，不宣称三任务已可训练，不生成新划分，也不把未知或ImageCHD补作阴性。

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

## 两例作者分类权重验收

2026-11-22获准批次使用完整MONAI SENet154、原输入1×1×85×215×215、CPU双线程、float32、0训练更新和0GPU。三个官方部署目录各提取model_1，均为裸state_dict、原生last_linear键，严格加载通过；旧fc映射分支未在本批触发。三任务各两次有限概率预测完成，未知参考不进入指标分母，方向和覆盖未核实状态保留。部署任务名不证明作者训练实验、种子、患者名单或父成员。

|任务|预测入口实测秒数（两例，含加载及报告，不含准入等待）|
|---|---:|
|aaoca_present|27.0542|
|origin_left|19.2176|
|high_risk|19.1027|

总流程1639.906秒，包含导入、预处理、提取及主存准入和阶段暂停；6次预测入口累计65.3745秒。50ms采样RSS峰2917855232B，低于8GiB限额；未测量操作系统精确峰值或显存峰值。主存不足时真实等待，余量满足后继续，100GB/整机80%保护未降低。结束清理13文件2068474130B，独立核查进程退出和临时根不存在；正式私有报告108620B，原ZIP保留。该记录只证明所选两例和三个单成员的功能流程，不证明诊断准确率、作者依赖版本逐值等价或其他模型成员兼容。

## 零更新检查点与输入拒绝验收

2026-11-22使用测试入口ef21d05，提取检测任务的一个真实作者成员，构建完整SENet154及真实AdamW/CosineAnnealingLR。生产save_checkpoint保存一次、weights_only加载一次；1144个模型张量逐值相等，模型严格加载、优化器参数组、调度器状态、Python/NumPy/Torch CPU后续随机输出及零次计数身份核对通过。恢复目标优化器和调度器从不同初始超参数建立。整个批次没有读取CT或诊断标签、执行forward、optimizer.step、scheduler.step、训练更新或GPU操作。

实测127.5秒，包含依赖导入、提取和阶段暂停；50ms采样RSS峰2226176000B。自动删除4文件1364493826B，独立核实进程退出、临时根不存在、原ZIP保留，正式私有报告1806B。检查点仅为同进程零更新序列化组件，真实AdamW moment为空；未验证成熟moment、训练消费位置、早停、跨进程train --resume或有监督训练续跑。

另用原配置diagnoses=None调用生产task_partitions及实际train入口，均在病例数组、模型和检查点创建前抛出预期ValueError；没有绕开监督门槛或补造标签。独立CPU随机状态内存往返耗时15.1376秒、零磁盘产物。上述局部检查不扩大正式训练许可或科学结论。
