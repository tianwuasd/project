# 共享验收阶段3接入

2026-11-18本轮唯一训练由助手2执行。助手3使用`run.py review --handoff PATH`只读生成已明确交接模型的预测步骤；加`--execute`且具有秘书release才执行。顺序为STU第一例/同目录两例接续、Q第一例/接续、完整模式下的Med第一例/接续，全部为生产预测命令，不包含train。

交接JSON由秘书确认阶段2结束后提供，字段如下：

| 字段 | 必需内容 |
| --- | --- |
| engine_root、engine_commit | 实际生产目录及最终预测完整SHA，须对应目录HEAD |
| review_mode、available_models、unavailable_models | complete需要三模型全部交接；partial须显式列出可用STU/Q及不可用模型→具体原因。models键必须与available_models完全一致，全部模型必须恰好归入一种状态 |
| python | 已有生产解释器 |
| simulator_root | D:/projects/test，仅读取既有源码和profile，调用纯虚拟资源决策场景 |
| roles | train=[ct_1001]、validation=[ct_1012]、test=[] |
| attempt_ledger | 阶段2唯一累计台账；工具仅读取、不领取、不结算。协议为chd-mini-attempt-v1，不能把另一计数协议自动转换为已验证证据 |
| work_root | D:/projects/back/t17/chd-review-20261118 |
| report_root | D:/projects/back/sync-reports/chd-review-20261118 |
| secretary_stage2_release | 秘书明确阶段2交接的文字记录，非空字符串 |
| cleanup_owner | 同轮共享模型/数组最后消费者的清理责任人 |
| source_images | 恰好ct_1001与ct_1012的真实原始CT路径，仅加载原图头；不能指向小区域副本 |
| models | 仅明确可用模型，每项包含config、prepared、evaluation_config、best_record、best_path_field；完整模式中Med包含prompt_mode及external模式的prompts路径。evaluation_config须为生产evaluate加载器兼容的实际配置，等待助手2交接 |
| models每项的training_commits | 不重复的实际训练完整SHA列表，保留首次/恢复及各阶段的全部实际来源，不以最终SHA重写旧证据 |
| models每项的version_reviews | 训练SHA为键；每项包含production_commit、approved_changed_files完整列表、secretary_review_reference。SHA不同时须有秘书最终逐项审核记录 |
| training_event_logs | 五阶段原始事件路径；保存后的update位置、显式优化器事件及台账分别保留，不按去重位置个数推断总真实更新次数 |

best_record为真实阶段2选模记录JSON，best_path_field为其中指向模型文件的字段名称，相对路径按记录所在目录解释。它应对应实际验证最佳模型，不能提供手工编造的记录。STU命令为`predict --config CONFIG --checkpoint best --preprocessed-source PREPARED --output OUTPUT --case CASE`，保留selected_best所得实际文件路径供核对最终加载来源，不将此文件路径传给STU枚举参数。Q/Med使用`models predict --config CONFIG --prepared PREPARED --checkpoint BEST_PATH --output OUTPUT --device cpu --case CASE`。Med同时转交真实提示模式和提示文件。生产CLI缺少显式--checkpoint时只读检查立即报告未接入，禁止自行改latest冒充best。配置必须是阶段2明确交接的本轮CPU技术配置。

工具对每个实际训练SHA执行`git diff --no-renames --name-only -z TRAINING FINAL --`，比较全仓库差异文件列表与秘书批准列表，必须完全一致；不按CPU补丁名称自动放行，不限缩目录隐藏其他差异。记录实际训练SHA、最终预测SHA、完整差异及秘书审核引用。即使文件差异为空但SHA不同，也需要对应审核记录。缺少最终审核或名单不一致时拒绝交接，不重新训练、不重置台账。此核对仅允许所列版本关系，不将旧训练证据转记为最终SHA训练通过。

部分模式仅为秘书明确交接的可用模型生成步骤，不为不可用模型读取配置、best路径或权重，不以官方预训练权重补充训练证据。报告始终INCOMPLETE，target_models保留stu/q/med，unavailable_models原样记录具体原因。Q某阶段有效更新与继续更新恢复分别记录，不能用一次更新代替恢复验证；增加训练尝试仍需独立授权。

AttemptLedger新增`limits()`，返回`stage_limits`与`total_limit`。无同名`.approval.json`时仍为每阶段2/总10。本轮秘书根据用户明确批准写入attempts.approval.json，严格绑定run_id与解析后的ledger_path，明确协议、全部字段及授权说明；仅接受q_base=3、其他阶段2、总11这一个例外。read/claim均核对批准文件及上限，原记录与outcome保持不变。助手2计算remaining时取该阶段上限减累计次数与总上限减全部次数的较小值，不能在调用方固定写成2/10。阶段3继续仅读台账，不能领取Q base第三次或Q localizer第三次。

第一步只请求ct_1001，第二步同目录请求ct_1001和ct_1012。工具核对第一例文件大小/修改时间没有变化，并要求第二步的真实复用/完成事件。几何调用生产resolve_preprocessed(storage.root,dataset=prepared.dataset,case_ids=[case],storage=prepared.storage)，原样传入显式所选storage，再用open(case,with_label=False)取得实际entry。记录source_plan、processed_spacing、axis_order、roi_xyz、原spacing、source_directory、所选指针及recipe_identity，不自行从根catalog默认重选配方。API按需映射所选数组，工具只提取元数据并释放entry，不读取体素或标签。该接口当前待助手2最终固定，适配仅静态检查。

核对实际entry的ROI与生产summary一致。预测输出应与真实原图头的shape/affine一致；按z顺序逐片读取预测标签实际dtype，核对ROI外零填充，不读取原始CT数值或SHA。mismatch明确失败。该范围证明原图空间头和ROI回填/零填充，计算仍仅ROI，不能证明全体积预测效果。

每个可用模型接续完成后，增加真实生产`evaluate --config EVALUATION_CONFIG --prepared PREPARED --predictions MODEL_OUTPUT --output OWN_EVALUATION.json`。Q同样使用生产公共evaluate入口，需要助手2交接其兼容评价配置；工具不自算Dice、不把预测summary当评价。保存实际评价JSON、病例/类别指标、technical范围、ROI和命令墙钟，必须覆盖两例。本轮仅完成该参数接入，真实评价待交接后运行。

build_review、每条预测/评价命令前及最终收尾均检查`git status --porcelain --untracked-files=all -- src run.py pyproject.toml`；存在未提交业务源码即拒绝执行，AGENTS私有说明修改不在业务检查范围。该检查只查Git源码状态，不扫描数据内容。

optimizer_evidence保留STU train事件、Q model_optimizer_completed显式优化器事件，并单列Q model_update_completed保存后位置；不去重后计为总真实更新次数。Med model_forward_backward与model_epoch_completed列为计算/epoch辅助事件，不能单独证明优化器成功更新。台账successful单列，总真实更新次数保持未核实，防止遗漏失败保存前的实际更新。

Q/Med的实际prediction run_dir随checkpoint/prepared/output/names/prompts/protocol变化，first和resume可以使用不同事件文件。每次生产summary须返回event_log绝对路径、start_offset及end_offset整数字节偏移。阶段3只读取该实际区间，检查文件大小、起止行边界和完整JSONL，保留源文件、区间、每条记录的字节偏移及原始记录。每步最多读取512KiB事件，超出即明确失败；不扫描旧固定日志或主console。每步原始生产summary先另存报告；起始步骤必须有ct_1001完成事件，接续步骤必须有ct_1001复用及ct_1012完成事件，缺失明确失败，mtime不独立构成通过条件。生产summary字段尚待助手2最终接入核实；本轮只准备适配，未执行预测。

进程使用已有pywin32实现Windows Job：主进程挂起创建，加入Job后恢复，全部后代继承Job；收尾终止Job并查询ActiveProcesses为零。该入口不在每批扫描全系统进程。Job内存上限16GiB；共享总额度和生产100GB/80%限制继续生效。该实现仅覆盖Windows，不能证明Linux/NCCL。

ReviewFinalizer在报告/台账/容量读取失败时仍进入独占目录清理。最终容量只记录，额度判断只在阶段执行前进行。仿真或进程启动前失败、尚无启动记录时可以清理；已尝试启动但缺少Job退出回执时保存retained_unconfirmed_workers和原始清单，等待确认后清理，不依据缺失回执推定worker已退出。清理只作用于阶段3独占目录，阶段2共享模型和数组保持由指定所有者管理。

已读取D:/projects/test的README、HANDOFF_OFFLINE_20261115.md、common.py与simulator.py。执行入口先调用工具既有simulation.check_resources，使用simulator.decision和实际生产gpu_decision，单列虚拟资源结果；不调用固定ct_1003的validate.py，不启动其中旧训练任务。阶段3真实模型执行仍等待秘书交接。助手3独占输出在核对Job退出后自动清理；共享模型和数组由cleanup_owner核对最后消费者后清理，助手3不能提前删除阶段2模型。
