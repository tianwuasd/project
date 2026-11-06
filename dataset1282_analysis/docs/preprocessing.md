# 方案 A 的 nnU-Net v2 预处理

当前确认的 HU/部分标签流程见 [0.3.0 工作流](hu-marginal-workflow.md)。以下保留历史试跑和逐例审核协议，不叠加到新的 `--workflow` 分支。

## 入口

```text
python run.py prepare --run scheme-a
python run.py prepare --run scheme-a --excel-pilot --research-python RESEARCH_PYTHON
python run.py prepare --run scheme-a --voxel-smoke --research-python RESEARCH_PYTHON
python run.py prepare --run scheme-a --review private-review.json --research-python RESEARCH_PYTHON
```

第一条使用原审计环境，只读清单和配置指定的 Excel。输出位于私有工作区的 `prepared/scheme-a/`。其余三条启动独立的 Python 3.11 数值进程，Windows 和 Linux 均传解释器路径。依赖 nnunetv2==2.8.1、对应 PyTorch、NiBabel、NumPy、SciPy、scikit-image、psutil。可在单独研究环境安装 `python -m pip install nnunetv2==2.8.1 nibabel psutil`。nnU-Net 导入前绑定本次运行的 raw、preprocessed 和 results 目录。只在 CPU 上运行，每次准入一例并检查内存和 CPU 余量。

## 患者分组

按 Excel 的 idx 严格关联原清单。先检查整表的 PatientID 与设备，再筛关联成功且未满 18 岁的扫描。LightSpeed VCT 和 SOMATOM Definition Flash 为两种来源，以患者最早儿童检查的设备、年龄层（未满 1 岁、1 至未满 6 岁、6 至未满 18 岁）分层，在层内用种子 20261104 和患者键的摘要排序，按 70/15/15 最大余数取整。重复扫描继承同一来源角色。留出 iCT 256 作主要目标、Brilliance iCT 作次要目标，跨来源和目标的 ID 从来源整体移出，只保留合格目标扫描作目标候选。双目标身份不清者待核查。Ultra、成人、缺关联、资格不明者各自登记。

`cohort.json` 及其摘要固定了本次分组和输入版本。私有文件含患者分组字段。候选人数按表内不同 ID 计。罕见病型分层、同人异号和术前资格需要额外来源。提交正式审核前核查病型分布与患者绑定；需要更正身份或调整设计时，应在使用冻结测试资料前重新版本化分组。训练、开发验证和两类测试分开使用。同来源测试与两个目标设备的图像及标签不会传给预处理进程。

## 运行方式

- `--excel-pilot` 按逐例 Excel 毫米采样试跑，仅访问已冻结的来源训练组。先核对严格关联、层数、行列尺寸、影像标签网格及原件摘要是否可用，生成私有 `excel-screening.json`。在网格相符的 native 候选中，对每台来源设备按病例键顺序取至多两名不同患者用于拟合。再按这些拟合病例的逐轴间距中位数与候选尺寸预估内存，在同设备其余患者中选预计内存最小且能容纳的一例，仅应用冻结计划。只应用组是本机容量测试样本。筛查记录保留预计超出预算的病例；正式网格和运行时内存准入由数值阶段重新确定。病例原始字节摘要在转换前再次校验。计划采用本批训练样本指纹，输出保存在独立的 `excel-pilot` 目录。此分支以数组 X/Y/Z 对应 Excel 列/行/层且拷贝保留原采样为待验证假设，使用局部零原点。输出可评价网格转换损失；患者坐标、正式全训练集规划及临床性能需要相应证据。见逐例 QC 和 `complete.json` 中的 `spacing_xyz`、`coordinate_frame` 与 `pilot_probe_cases`。

- `--voxel-smoke` 从两台来源的训练候选池按病例键选至多两例、从开发候选池选至多一例，仅接受 native 且实读头矩阵一致的文件，每组内患者不同。索引网格的单位为一个数组步长，输出单独存放，只用于检查工程链路。开发池没有合格样例时明确记录 0。
- `--review` 读取经审核的私有输入。每例需核实术前状态、患者绑定、完整监督范围、影像标签逐体素对应以及原始数组下标到 RAS 毫米空间的完整仿射矩阵。原点冲突只能在有证据确认两个数组的**同一下标对应同一位置**时进入。需要移动标签、重建索引或存在部分未知监督的病例先留待核实。对训练及开发均要求非空，开发患者只取最早合格扫描，同日按研究编号。签核者按当前清单摘要和两个原始文件摘要登记证据。预处理保留 0–10 数字标签。

审核文件的结构见下方示意。实际 `cases` 对象需要同时包含合格训练及开发病例，此处仅展现单条。矩阵必须来自该例校准，下面为合成示例。

```json
{
  "cohort_digest": "<prepare 输出的摘要>",
  "intensity": "body_percentile",
  "cases": {
    "SYNTHETIC_CASE": {
      "source_sha256": {"image": "<sha256>", "label": "<sha256>"},
      "evidence": "<私有逐例配对、毫米标定、术前、患者及标注签核记录>",
      "affine_ras_mm": [[0,-0.8,0,40],[0.7,0,0,10],[0,0,1.5,-2],[0,0,0,1]],
      "patient_group_verified": true,
      "index_alignment_verified": true,
      "preoperative_verified": true,
      "full_supervision_verified": true
    }
  }
}
```

### 从转换 Excel 读取逐例间距

审核已确认当前数组 X/Y/Z 分别对应原列、原行、原层且保留原采样时，可在该病例的审核条目中将 `affine_ras_mm` 替换为以下字段。程序按来源键唯一关联该例 Excel 行，核对实际 NIfTI 尺寸与 Cols、Rows、valid_z，然后读取 PixelSpacingX、PixelSpacingY、calc_zSpacing（mm）。SliceThickness 是层厚，缺少层间距时不将其代入。原点和方向需有独立依据。此例展示字段格式，方向矩阵每一列对应一个数组正轴在 RAS 中的单位方向。

```json
{
  "spacing_source": "excel",
  "excel_grid_verified": true,
  "directions_ras": [[0,-1,0],[1,0,0],[0,0,1]],
  "origin_ras_mm": [32,1,5]
}
```

输入间距、Excel 行号及推导出的矩阵保存在私有 request.json。尺寸冲突、间距缺失或非正的病例会报错。训练共同目标间距仍由 nnU-Net 根据已核实的训练病例规划。

## 自动规划和回写

1. 对入选文件核对两份原始 SHA256。只在衍生副本使用校准坐标。默认灰度分支对每幅 CT 独立计算体内近似区域的 0.5 和 99.5 百分位。实现为低分辨率 RAS 图像的 Otsu 阈值、最大连通区和逐轴位填孔，不使用标签，用近似区域估计灰度范围后映射整幅图。零背景保留，低于窗限的原非零值保留为 0.001，避免裁剪误删。体内区域与裁剪覆盖需在来源病例做视觉复核。已核实 HU 的全批病例可统一设 `intensity=verified_hu`，每例另需 `hu_verified=true`，随后由官方 CTNormalization 使用训练前景确定共同的裁剪、均值和标准差。
2. 官方 DatasetFingerprintExtractor 只接收来源训练病例。总前景抽样数为 min(1 亿, 10 万 × 训练例数)，故小数据量运行低于官方默认总上限。ExperimentPlanner 自动估计目标间距、轴序、patch 和配置。采用 3d_fullres、NibabelIOWithReorient 和 DefaultPreprocessor。未知 HU 分支在读入前已映射，规划通道使用 noNorm。
3. 对选定训练和开发病例用同一冻结计划完成非零裁剪、影像与离散标签重采样、前景采样位置生成。用官方读取器回读压缩数组。将处理后标签按体素中心最近邻恢复至裁剪前网格，对十个编号分别记录原始体素数、回写体素数及 Dice。该指标检验预处理对已有标签的影响。细血管分支存留及结构名称仍须人工核验。
4. `physical/preprocessed/Dataset282_CHDSource/` 内有 dataset_fingerprint.json、nnUNetPlans.json、3d_fullres 压缩病例、开发组原网格标签和 splits_final.json。后续训练使用该文件的 fold 0。原始目录中的 dataset.json 仅列训练病例，冻结计划后直接复用产物。后续推理在未知 HU 分支也须先对输入 CT 调用同一 `map_intensity`。训练实验臂共用本次分组、数据和计划，模型选择使用开发组。
5. 每个数值运行保存 request.json、preparation.log、逐例 qc 和最后写入的 complete.json。只有带完成标记的目录可交接。重跑相同的分组检查会核对冻结内容；数值运行已有输出时请使用新 run 名。原件、患者清单、审核证据及缓存始终保存在本地工作区。

官方方法见 [nnU-Net 实验规划与预处理](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)。
