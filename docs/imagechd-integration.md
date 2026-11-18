# ImageCHD技术集成入口

工具分支为`codex/chd-assistant3-imagechd-acceptance-20261117`。schema 3走`integration.py`，schema 2保留只读配置检查；新CLI不会启动旧四例工作流。生产源码和smoke均由原负责人维护。

## 配置与实际命令

私有配置包含schema=3、profile=technical、engine_root、完整engine_commit、python、work_root、report_root、roles、intensity_mode及steps。相对目录按该配置所在目录解析。roles只允许train、validation、test；本轮train为ct_1001，validation为ct_1012，test为空。总病例最多2，技术配置不得声明独立测试集。

每个step包含唯一name、kind（prepare/preprocess/train/predict/evaluate）、args字符串列表、timeout_seconds（最多1200）和receipt（报告目录内相对JSON路径）。train还需要stage（stu/q_base/q_localizer/q_local/med_sequence）及attempt_protocol=chd-mini-attempt-v1；predict需要checkpoint=best。args直接传给外部生产run.py，不经shell；其中{work}和{reports}替换为本次目录。真实参数应在助手2最终CLI发布后填写，当前不提供声称兼容的预测命令。

上游参数必须包括当前technical.stage_roles、实际来源路径、best选择和停止/恢复位置；模型默认预算、早停与选模消费助手2resolve_budget结果。参数配置不能自行覆盖为完整正式训练。生产回执至少包含state=completed及实际roles；训练回执提供restored与best_path，预测回执提供model（stu/q/med）、checkpoint、resumed_cases_skipped及source_geometry_restored。字段应从实际事件提取，不能用配置意图生成。计时、几何、资源和消费证据随回执提供。TECHNICAL_COMPLETED需要五阶段成功更新/恢复/best以及三模型best逐例接续和几何证据；缺项保留INCOMPLETE。完整计划其余条目仍逐项审核实际记录。

## 累计尝试接入约定

此约定是工具提供的接口，生产层接入及真实验证仍待助手2最终版本。每个训练阶段包括首次与恢复最多两次尝试，五阶段总共最多十次。一次尝试覆盖一次计划更新和全部梯度累积；Q accumulation=4时最多八个训练微批次/阶段。多rank由主rank领取一次并广播，其他rank不重复领取。

监督器传递CHD_MINI_ATTEMPT_LEDGER、CHD_MINI_STAGE、CHD_MINI_TOOL_ROOT和CHD_MINI_ROLES。生产调用方在开始构建训练输入之前调用工具AttemptLedger.claim(stage)，完成后调用finish(stage, number, outcome)，outcome为successful/skipped/failed/interrupted。可导入工具src目录，或调用run.py attempt --ledger PATH --stage STAGE；返回number后执行训练，再用--number NUMBER --outcome OUTCOME结算。不要用子进程退出成功代替优化器更新成功。实际计算错误继续抛出，监督器确认所属进程退出后将未结算尝试记为interrupted。

记录通过文件锁和原子JSON写入保存。恢复旧检查点不能减小记录次数，运行目录改变不能重置本轮额度。当前入口拒绝在已有累计记录的report_root重新开始；正常保存恢复应编入同一次运行的不同步骤。异常重启如需保留模型继续，须先核对归属、原始累计记录及保留授权，再接入恢复入口；当前失败收尾自动清理独占工作文件，不能声明异常模型恢复已经支持。

## 产物批准及检查

inspect --config PATH只读检查schema 3、步骤和生产版本。run --config PATH --run-id ID --approval PATH执行已批准步骤。approval为私有JSON，包含approval_reference、settings（inspect返回的完整规范化settings）、max_files、max_bytes、max_report_bytes。该文件记录秘书取得的实际许可；工具不会把估算数值转成许可。

work和reports必须独立于生产目录且互不包含。共享数组和已有权重仅原地引用，不能放进将被清理的work。当前同时存活文件在阶段边界盘点，无逐批全目录扫描；日志有容量上限，实际文件错误和超限会终止执行。该盘点不保证拦截阶段内部瞬时磁盘峰值，步骤所需空间必须事先计算和批准。资源与显存数据来自公共层回执；工具未测量的数据保持未测量。

执行结束或失败后确认所属进程退出、核验完整路径及目录清单，再删除本次work，保留累计记录、命令日志、summary和cleanup。当前步骤命令不得写共享输入或使用报告目录保存模型。真实三模型验收、CUDA/NCCL、正式128³及患者独立评价均等待各自运行证据。
