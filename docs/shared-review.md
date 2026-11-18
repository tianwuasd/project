# 共享验收阶段3接入

2026-11-18本轮唯一训练由助手2执行。助手3使用`run.py review --handoff PATH`只读生成六个确定步骤；加`--execute`才执行。顺序为STU第一例/同目录两例接续，Q第一例/接续，Med第一例/接续，全部为生产预测命令，不包含train。

交接JSON由秘书确认阶段2结束后提供，字段如下：

| 字段 | 必需内容 |
| --- | --- |
| engine_root、engine_commit、training_engine_commit | 实际生产目录及完整SHA；版本不相同时停止本次路线，旧事件保留原版本 |
| python | 已有生产解释器 |
| simulator_root | D:/projects/test，仅读取既有源码和profile，调用纯虚拟资源决策场景 |
| roles | train=[ct_1001]、validation=[ct_1012]、test=[] |
| attempt_ledger | 阶段2唯一累计台账；工具仅读取、不领取、不结算。协议为chd-mini-attempt-v1，不能把另一计数协议自动转换为已验证证据 |
| work_root | D:/projects/back/t17/chd-review-20261118 |
| report_root | D:/projects/back/sync-reports/chd-review-20261118 |
| secretary_stage2_release | 秘书明确阶段2交接的文字记录，非空字符串 |
| cleanup_owner | 同轮共享模型/数组最后消费者的清理责任人 |
| source_images | 恰好ct_1001与ct_1012的原地影像路径，用nibabel读取头部进行输出几何评价 |
| models | stu/q/med三项，每项包含config、best_record、best_path_field、event_logs；Q/Med还包含prepared；Med包含prompt_mode及external模式的prompts路径 |

best_record为真实阶段2选模记录JSON，best_path_field为其中指向模型文件的字段名称，相对路径按记录所在目录解释。它应对应实际验证最佳模型，不能提供手工编造的记录。STU命令为`predict --config CONFIG --checkpoint BEST --output OUTPUT --case CASE`；Q/Med使用`models predict --config CONFIG --prepared PREPARED --checkpoint BEST --output OUTPUT --device cpu --case CASE`。Med同时转交真实提示模式和提示文件。生产CLI缺少显式--checkpoint时只读检查立即报告未接入，禁止自行改latest冒充best。配置必须是阶段2明确交接的本轮CPU技术配置。

第一步只请求ct_1001，第二步同目录请求ct_1001和ct_1012。工具核对第一例文件大小/修改时间没有变化，提取第二步新增原始事件中的prediction_reused或model_case_reused，并用源图和输出的真实shape/affine核对几何；不从配置生成restored或geometry=True。原始production summary随结果保留。当前逐项完整验收仍输出INCOMPLETE，交由最终审核根据真实记录列明缺项。

进程使用已有pywin32实现Windows Job：主进程挂起创建，加入Job后恢复，全部后代继承Job；收尾终止Job并查询ActiveProcesses为零。该入口不在每批扫描全系统进程。Job内存上限16GiB；共享总额度和生产100GB/80%限制继续生效。该实现仅覆盖Windows，不能证明Linux/NCCL。

已读取D:/projects/test的README、HANDOFF_OFFLINE_20261115.md、common.py与simulator.py。执行入口先调用工具既有simulation.check_resources，使用simulator.decision和实际生产gpu_decision，单列虚拟资源结果；不调用固定ct_1003的validate.py，不启动其中旧训练任务。阶段3真实模型执行仍等待秘书交接。助手3独占输出在核对Job退出后自动清理；共享模型和数组由cleanup_owner核对最后消费者后清理，助手3不能提前删除阶段2模型。
