# AAOCA 强度与空间预处理工作流

本模块处理原始 HU NIfTI：**RAS 轴重排 → HU 线性重采样 → clip[-1000,1000]/1000 → float32 [-1,1]**。全扫描范围保留，源文件只读。依赖安装、数据格式和基础接口见 [PREPROCESSING.md](PREPROCESSING.md)。

## 自动间距与 nnU-Net 的关系

nnU-Net 的默认规划器根据训练集间距中位数确定全分辨率目标间距，并在间距和体素数同时高度各向异性时调整最粗轴；还会规划输入块、批量大小及网络。本模块借用其**间距选择思路**，以完整扫描范围的 RAS 头信息计算建议，不执行 nnU-Net 的裁剪、监督式前景统计或网络规划。

因此这里的 `auto` 不是正式 `nnUNetPlans.json`，也不证明该间距对 AAOCA 任务最优。不同原始分辨率仍可能产生上采样或下采样；自动间距也不保证各向同性。0.5 mm 只是可手动选择的参数，没有默认推荐地位。

建议只在训练集清单上估计自动间距，然后把确定的三个数值用于验证/测试数据，不在每个子集重新拟合。规划报告包含参与体积数、间距分位数、是否触发各向异性调整及完整视野的说明。未识别的空间单位或冲突头信息会中止自动建议，不静默排除后继续拟合。

nnU-Net 默认 CT 标准化与固定 HU 映射不同。若后续决定使用标准 nnU-Net 训练，应从保留的原始 HU 影像让它统一规划；若必须使用这里的归一化结果，需要另行配置匹配的归一化与重采样策略，避免把 [-1,1] 再当 HU 做 CT 标准化或反复插值。本模块输出不是可直接替代 nnU-Net 预处理缓存的格式。

参考：[官方间距规划实现](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py)、[官方计划文件说明](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation_plans_files.md)。

## 查看方案，再运行

以下 `/path/to/...` 都是需要替换的示例。命令行间距按 RAS 轴顺序；不会默认采用 0.5 mm。

```bash
# 只查询当前机器与输入目录
aaoca-preprocess doctor --path /path/to/AAOCA --path /path/to/results-parent

# 只读：自动建议，以及按该建议计算的每例网格/容量/内存
aaoca-preprocess suggest-spacing --input /path/to/AAOCA
aaoca-preprocess plan --input /path/to/AAOCA --spacing auto

# 显式选择 auto 后，前台执行归一化与重采样
aaoca-preprocess prepare --input /path/to/AAOCA \
  --output /path/to/preprocessed-v1 --spacing auto
```

也可将 `--spacing auto` 换为明确数值，例如 `--spacing 0.6 0.6 0.3`；示例数值不代表适合所有数据。`plan` 只读头信息，不能确认像素完整性或 HU 来源。正式执行保留前后源哈希和逐体素输出校验。

## CPU 与内存调度

| 参数 | 默认 | 作用 |
|---|---:|---|
| `--workers` | 1 | 最大并发病例数，实际受 CPU 与内存预算约束 |
| `--compute-threads` | 1 | 每例同时插值的块数；不会一次排队所有块 |
| `--read-workers` | 1 | 同时进行哈希、完整解压、写出或校验的流数 |
| `--chunk-depth` | 8 | 每个计算块的输出切片数 |
| `--max-case-gib` | 8 | 单例预计工作集上限；超出单例失败 |
| `--memory-budget-gib` | 未指定 | 整批内存预算；仍受当前可用内存 75% 限制 |
| `--compression-level` | 1 | gzip 0–9；改变大小/耗时，不改变体素值 |
| `--scratch` | OUTPUT/.scratch | 临时解压和磁盘数组所在目录 |

CPU 上限取进程 affinity 与可见 cgroup 限额的交集，并检查 cgroup 父级；用户仍需遵守服务器实际分配。线程数不能超出可见 CPU 预算。调度按正在运行病例的预计内存之和决定是否接受下一例，避免所有病例按最大病例估算导致无谓停机。

内存数值是保守估算，包含源数组、并行块和输出映射页；不是硬 RSS 上限。源数组仍需整体加载，磁盘映射也可能占文件系统缓存。磁盘检查不包含个人配额，服务器运行前需核实。资源设置不会改变固定 HU/空间公式；压缩等级、线程与块大小改变后仍可复用数值配置相同且校验通过的结果。

```bash
# 会实际处理最多3个按体积大小选取的样例；不是只读命令。
# 不同 trial 独立保存结果，比较逐体素摘要和实际时间。
aaoca-preprocess benchmark --input /path/to/AAOCA --spacing auto \
  --output /path/to/private-benchmark-new --samples 3 --worker-options 1 2 4
```

`benchmark.json` 保存实际并发、墙钟时间、CPU 时间、抽样峰值进程 RSS、每例阶段耗时及数值一致性。数据规模、缓存、压缩比、磁盘和任务顺序都影响结果；本机合成测试不能推导服务器加速倍数。测试目录必须新建，保留试验产物供核对，不自动清理历史结果。

## 后台运行、停止与续跑

```bash
aaoca-preprocess start --input /path/to/AAOCA \
  --output /path/to/preprocessed-job-v1 --spacing auto \
  --workers 2 --compute-threads 2 --read-workers 1

aaoca-preprocess status --output /path/to/preprocessed-job-v1
aaoca-preprocess stop --output /path/to/preprocessed-job-v1
# 确认 worker 已退出后，按创建任务时的参数接续
aaoca-preprocess resume --output /path/to/preprocessed-job-v1
```

`start` 使用当前解释器启动独立进程，在输出旁建立 `preprocessed-job-v1.runtime/`。其中 `job.json` 冻结输入列表、源大小/修改时间、参数、自动间距依据和实现签名；`process.json` 保存 PID/创建时间；`worker.json` 保存本次执行状态；`worker.log` 追加病例日志。新文件加入输入目录不会悄悄加入已冻结任务。每例处理仍校验原始清单提供的 SHA256，并在处理前后重新哈希。

`status` 将进程身份、worker 状态与最后一次进度分别呈现；旧快照为 running 不等于进程存活。`progress.json` 记录活动病例的阶段与已计算切片数，计算阶段最多每2秒落盘一次，不伪造剩余时间。

`stop` 写入协作停止请求，不向 PID 发送强杀信号。解压/哈希/插值/校验会在检查点停止；gzip 写出需等当前保存返回。已发布成果保留，中途病例下次重算。再次 `resume` 复用冻结列表和配置，拒绝改变的源时间/大小、代码或依赖。更新代码后应使用新输出与任务目录。不同环境使用同一结果目录会触发签名冲突，不能靠删除锁文件绕过。

可用 `AAOCA_PREPROCESS_PYTHON=/path/to/env/bin/python bash /path/to/project/AAOCA/preprocess_server.sh start ...` 调用源码入口。启动器和后台 worker 默认将未设置的数值库线程变量设为1；既有设置会保留。无需 GPU/CUDA。

## 方法与工程核对

对照仓库内 [medical-image-preprocessing](../medical-image-preprocessing/PREPROCESSING_MAP.md) 的流程拆分：

| 核对点 | 当前实现和边界 |
|---|---|
| 强度单位 | NIfTI 缩放应用一次，输入由提供方保证 HU；先 HU 插值后截断缩放 |
| 空间信息 | 校验毫米、有效 affine、qform/sform；RAS 重排保留斜切/剪切 |
| 重采样视野 | 整数尺寸取整后保留体素单元视野和中心；记录实际间距 |
| 可追溯坐标 | 保存原始到输出、输出到原始的下标映射；不是逆向恢复原像素 |
| 下采样 | 报告降采样轴/尺寸倍率；沿用线性内核，无抗混叠滤波；需任务质量验证 |
| 处理范围 | 全扫描，不自动裁剪、去噪、配准、修片、分割或训练 |
| 完整性 | 源前后哈希、全像素回读、gzip CRC、最小化输出头、原子发布 |
| 运行工程 | 资源预检、独立病例、冻结后台任务、阶段进度、停止/续跑、实测基准 |

这一发布完成的是软件封装与合成验证。服务器部署、真实病例质量检查和全量运行应以各自的日志与结果为证据，不能由代码测试替代。
