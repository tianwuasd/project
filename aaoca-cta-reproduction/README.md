# Shiri AAOCA CCTA 研究准备

研究对象：[Shiri等，Nature Communications 16:3095 (2025)](https://doi.org/10.1038/s41467-025-58362-9)。作者发布版本固定为 `b134e621943f60f20caf6a9272aa85ceeeb441f8`，来源与非商业许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

已验收输入冻结、患者隔离、私有输入迁移，以及r2两例作者原索引裁剪/真实TorchIO0.19.6分类重采样/量化/轴序/几何/复用。r2与固定作者直接计算一致，28.031秒完成并清理。三任务作者model_1已各严格加载并完成两例CPU预测，共6次；未知参考均不计入性能。零更新检查点的同进程磁盘保存加载、模型张量、优化器参数组、调度器及CPU随机状态恢复已通过。有更新的训练恢复、CUDA、五成员集成和诊断性能尚未验收。现有分割只用于定位，未知诊断保持为空。

## 入口

```bash
python -B setup_env.py
python -B run.py audit --config project.local.json
python -B run.py check-training --config project.local.json
python -B run.py export-inputs --config project.local.json
python -B verify_inputs.py --config project.local.json
python -B run.py --code-info
python -B run.py prepare-dataset --config <bound-task-config.json> --check
```

`audit`显式生成或核对分组，已有不同清单时停止。`prepare --case`和`check-training`直接读取冻结清单，不重新遍历全部病例。`prepare`只核对所选病例的状态与几何，同源同方法且输出状态相符时复用。`smoke.py`限私有配置中的两例，零训练更新，结束自动清理本轮产物。

`export-inputs`生成私有相对路径清单及服务器配置。`bind-inputs`在目标机器显式核对路径与大小、登记目标机器状态，不改变分组；已有不同配置或清单时停止。大小检查不等于文件内容哈希校验。私有配置和清单不公开。

`organize-two-tasks`组织独立患者清单，`bind-dataset`使用既有真实数据根生成两个服务器任务配置。`prepare-dataset`按完整清单串行完成本工程作者分类预处理；检测/侧别共享同一data_root结果，完成病例复用，接续对本批已知未完成文件核对归属后处理。STU-Net标准化数组不能直接作为本分类器输入，原CT与定位标签继续复用；操作与已测边界见服务器说明。

定位文件缺失、明确命名的LV/AO缺失或结构为空时，分类预处理在该例目录写入`localization-skipped.json`并继续后续病例；网格不匹配、输入变化、文件损坏和资源错误仍直接停止。配置、冻结分组、完整分类回执及模型预处理方法保持原有含义。进度分别记录`completed_count`、`skipped_count`、`remaining`，全部病例处理后`state=completed`；完成数量与跳过数量之和等于计划数量。训练、标签可用性检查和预测共同读取源身份匹配的排除记录，报告实际样本数；跳过病例不进入指标分母，也不补入其他分组。原标签补齐或变化后须核对该例来源与排除记录再重新预处理，禁止修改完整结果身份绕过复用检查。

## 资源与验收

定位使用整例解剖标签，距离计算缩到结构区域，原CT只读取裁剪范围。阶段估算依据标签头/实际bbox/ROI及重采样数组，源码保留100GB/整机80%约束。各数据保留源数组方向，索引偏移的物理方向不同，统一分类输入方向尚未验证。实际已验和未验边界见[能力审核](docs/CAPABILITIES.md)。

新增路径集中管理线程、数据供给及单个显式CPU/CUDA设备；两例分类预测已执行主存准入、真实余量等待和50ms采样，超限终止分支未触发。没有自动扩卡、分布式分类或等价大块处理验收。正式训练要求经审核方向及开口/目标近端覆盖；探索性预测允许明确的author_native_index配方保留未核实状态，记录真实几何且不宣称论文效果或临床有效性，不自动把输入转成RAS。新增接口与运行前条件见[分类模块](docs/CLASSIFICATION.md)。

两例真实定位裁剪及复用短流程总14.13秒，零训练更新；4份临时文件约16.1MiB已清理。2026-11-22两例三任务预测入口累计65.3745秒，总流程1639.906秒包含主存及阶段等待，采样RSS峰约2.72GiB；13份副产物2068474130B已清理。计时来自Windows本地，不能作为全量或服务器速度证据；训练速度未测量。

## 文件与服务器

- `src/aaoca_reproduction/`：输入冻结、定位裁剪、迁移引用、资源准入与时序记录。
- `configs/zmic.example.json`：公开服务器配置示例，不含真实病例编号或诊断。
- `update.py`：本项目自带完整更新核心，只同步本独立目录，固定Git提交并核验内容；仅使用Python标准库与Git。
- `private/`、`work/`、`reports/`：私有运行内容，排除Git；`state_root`可放course，相对路径按配置文件目录解释。
- [服务器部署与上传](docs/SERVER.md)：最小上传清单、现有环境及入口。

作者权重ZIP已取得，三个任务各一个部署成员的两例CPU功能流程通过；现有MONAI1.5.2复用。诊断读取兼容confirmed、document_explicit及research_sources研究策略，直接读取整理工作簿data页，各任务独立筛选并保留来源、未知及CT匹配状态；用户明确确认的阴性另保存逐例user_attestation，不伪造医生确认。全体CT登记映射不以旧20名冻结队列为纳入上限，正式训练仍需所选任务两类与CT匹配、经审核方向及覆盖。完整论文复现、正式训练和有更新恢复未验收，新增模型计算、训练及权重下载须取得各自具体许可。
