# HU 与部分标签预处理（0.3.0）

## 调用与环境

使用独立审计环境调用 `setup_env.py --check` 和本项目入口。数值进程使用独立 Python 3.11、nnunetv2==2.8.1、PyTorch、nibabel、scipy、scikit-image、psutil；不修改安装包源码。

```text
python run.py prepare --run hu-source-v1 --workflow /private/policy.json --frozen-cohort /private/cohort.json --research-python /research/bin/python --workers 2
python run.py prepare --run hu-source-v1 --workflow /private/policy.json --frozen-cohort /private/cohort.json --research-python /research/bin/python --workers 2 --resume
```

Windows 的研究解释器用 `环境目录/Scripts/python.exe`。原件与输出位置由 `project.local.json` 或 `--config` 提供。`--case` 可重复指定技术子集，明确标记 `technical_subset`；不指定则处理全部合格来源训练和开发患者的索引扫描。测试影像不传给本阶段数值进程，后续应通过独立冻结计划应用入口接入。

## 数据与数值流程

1. 校验主清单、Excel、原冻结分组，应用有原件摘要约束的明确别名和排除项。别名只接受唯一精确患者编号，不能靠模糊姓名。原有患者角色不重排。每个角色内采用排除后的最早合格扫描。
2. 原件 SHA256、三维尺寸、Excel Cols/Rows/valid_z 必须相符。毫米间距取 PixelSpacingX、PixelSpacingY、calc_zSpacing，不用 SliceThickness 补缺。
3. 仅处理已确认的单位阵或标签固定 +1 原点占位头。工作副本使用同一 Excel 毫米网格，局部原点为零，数组不移动。该网格不声称恢复真实患者坐标。
4. 原 NIfTI 读出的缩放值直接按 HU 使用，额外比例 1、偏移 0。这是研究假设，不是 DICOM 校准验证；不叠加旧版体区百分位映射。保存原头、新头、原件摘要和假设。
5. 仅训练病例参与官方 DatasetFingerprintExtractor 和 ExperimentPlanner。正式运行保留官方总计 1 亿前景抽样；技术子集采用 `min(1 亿, 10 万 × 训练例数)`。缺标签可能使强度抽样偏向已标结构，须在实验中报告这一限制。
6. 官方 CTNormalization 用训练前景统计裁剪并标准化；官方 3d_fullres 计划自动决定间距、轴序和 patch。DefaultPreprocessor 对图像和离散标签实施各自的官方重采样。开发集只应用冻结计划。
7. 从缓存回写原数组网格，检查有限值、形状、各类体素数与 Dice，阻止原本存在的类别全部丢失。回写 Dice 是网格转换检查，不是模型分割准确率。
8. 输出 fold 0 划分、逐例监督元数据及训练输入合同。不会自动开始完整科研训练。

## 边际监督

全体积未出现、又没有真实缺如复核的类别记为 unknown。该规则不能识别已经出现的结构内部是否仍有局部漏标；此情况需要区域级监督定义。

11 通道 softmax 中，标签 0 的允许集合为 `{0} ∪ unknown`，边际交叉熵使用集合概率之和；已标前景仍为单类目标。Dice 在相同合并空间计算，包含合并背景组，unknown 不作为独立空类别惩罚。私有 `known_absent_classes` 可逐例声明已复核真实缺如，保持普通负监督。损失忽略 -1 目标。

`training/trainer.py` 中的 `nnUNetTrainerMarginal` 使用官方 dataloader 保留的 batch `keys` 绑定元数据，支持深监督并拒绝缺失元数据。创建 Trainer 时核验计划、划分及监督元数据摘要。验证指标仅累计该类别有监督的病例。

本版提供单设备接口，DDP 明确拒绝，不能套用全监督归约。默认全体积验证也被阻止，避免把缺标签算成空真值；正式全体积部分标签评估需后续接入。CPU 短测用小网络验证 batch、梯度和更新，不代表完整 nnU-Net 科研训练。

## V2 调度与恢复

机制参照 server-engineering V2 / VISTA 0.7.2，源提交 `47016371e8d56a19630585f49c2cc0299259fa34`，技能包 SHA256 `67dd1fe2529cfbe8dd2dddf9d5c19e1b479a6d5971f7325d9ea4d431d8aeac67`。`vendor/vista_read_admission.py` 来自该版 `05_src/read_admission.py`，适配 JSON 导入；资源快照沿用 `vista_live_resources.py`。队列为本项目实现，未移植 B0 算法或 GPU allocator。

- CPU/RAM 上限均为可用资源的 80%，至少预留 4 GiB 和阶段初始可用内存的 20%。每个进程一个计算线程，进程上限由 `--workers` 指定。
- 冻结阶段预算，每次放行重读资源。只有身份匹配进程的 USS 才能抵扣已兑现的预留内存。未知容量时不放行。
- 读取用本机临时目录的操作系统文件锁；读取结束释放，计算不占读取槽，进程死亡自动释放槽。
- 约每秒检查内存。压力触发时只停止本次进程，保留已提交结果，降低并发和内存预算，最多恢复 3 次，降低后的预算传给后续阶段。单例仍超容量时有界等待后明确停止。
- 无压缩 `.nii` 工作副本、标签分块统计、隔离进程和按病例缓存减少重复工作，不更改数值算法。`--staged-extension .nii.gz` 可用于同任务基线比较。
- `status.json` 包含阶段、完成数、活动任务、最近进度年龄、耗时、预算和峰值 RSS。`events.jsonl` / `run.log` 记录准入、完成、压力和失败；`worker_logs/` 保存数值细节。
- 数值结果先写独立临时目录，核验后替换目标，最后提交所有缓存文件摘要。恢复必须匹配代码/输入/政策、数值依赖版本和官方核心模块摘要，重新核验原件和已提交文件。换研究环境后不能混用旧缓存。缺失完成记录可以重跑；已有记录但内容损坏会拒绝复用。
- 工程锁阻止更新和计算相互覆盖。同一工程的验收命令应顺序运行，不能删除锁绕过。

## 可靠短测

使用审计环境调用：

```text
python scripts/short_test.py --work-root /private/acceptance --research-python /research/bin/python
```

覆盖精确别名、排除、冻结角色、重复患者、+1 占位头、缓存损坏和内存压力降级；边际 CE 与普通 CE 一致性、unknown 概率重分配不变性、深监督梯度及忽略标签；官方串行/并发预处理、逐体素一致性、原件摘要、真实 dataloader/Trainer 更新、纯复用、损坏拒绝、未提交任务重跑后的结果一致性。

真实验收追加 `--real-config /private/project.json --policy /private/policy.json --frozen-cohort /private/cohort.json --case CASE`，需包含部分标签病例和训练/开发两种角色。输出私有 `checks.json`、`numerical.json`、`recovered.json`、`report.json` 及日志。同一病例集比较单进程 gzip 工作副本与双进程无压缩副本，实际速度不能外推为全量或服务器速度。

参考：[nnU-Net v2.8.1](https://github.com/MIC-DKFZ/nnUNet/tree/v2.8.1)、[部分标签边际损失研究](https://doi.org/10.1016/j.media.2021.101979)。当前 Dice 是项目的显式合并空间实现，不声称复现该论文完整训练方案。
