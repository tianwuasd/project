# 本地医学影像预处理项目盘点

本目录记录 2026-10-13 对 `code_project/` 的一次离线盘点。`inventory.json` 记录项目与来源，`file-manifest.json` 逐文件记录来源、导出位置、SHA-256、许可证与变换，`export-review.json` 记录未导出的例外。路径以 `code_project/` 为逻辑根，不含本机盘符。已选出的自有预处理代码放在 `local-code/`，附带许可证据的第三方代码放在 `vendor/`，许可未明或不适于此处公开分发的摘录留在 `local-only/projects/`。这里只收录源码摘录和文档，不能当作完整可安装项目或数据集。真实影像、病例资料、模型权重与运行产物均未纳入。

## 从哪里读起

| 问题或场景 | 优先看的入口 | 处理原则 |
| --- | --- | --- |
| DICOM CT 转 NIfTI，保留 HU 和物理几何 | `projects/local-code/aaoca/convert_ct.py`、`common.py`；公开仓库中另有 [AAOCA 目录](https://github.com/tianwuasd/project/tree/714a0a5b9ec5fdea605cebe52c019844dbfa7855/AAOCA) | 按物理位置排序、检查层距与方向，应用 RescaleSlope/Intercept，再将 LPS 仿射转为 RAS；不补缺片，也不自动脱敏。 |
| DICOM 超声帧与时间信息 | `projects/local-code/aaoca/convert_us.py` | 保存帧和时间、区域校准信息；与 CT 的三维重采样不是同一问题。 |
| DICOM 入库前审查或去标识风险检查 | `projects/vendor/dicom-audit-tools/skills/pydicom/scripts/`，尤其 `dicom_inventory.py`、`transfer_syntax_inspector.py`、`deidentification_audit.py` | 先清点传输语法、帧数和标识符，再决定能否解码或去标识；审查不能证明像素内没有烧录信息。 |
| 像素 LUT、窗宽窗位、颜色转换和帧解码细节 | `projects/vendor/pydicom/pydicom/pixels/processing.py`、`utils.py` | 这是 pydicom 3.0.2 的部分源码，用来查处理语义；缺少整包依赖，不能直接当安装包。 |
| NIfTI 图像与标签几何质控 | `projects/local-code/whole-heart-qc/medical.py` | 只读检查 affine、间距、单位、标签和非有限值，先发现错位与异常，再进行训练准备。 |
| whole-heart 标签、病例划分、nnU-Net 数据准备 | `projects/local-code/whole-heart-pipeline/05_src/data_pipeline.py`、`prepare_nnunet.py` | 官方标签号与训练标签号双向映射；来源训练集计算指纹和规划，目标组不参与拟合。后者调用 nnU-Net 官方 API。 |
| ImageCHD/CHD CT 复现实验 | `projects/local-code/chd-ct/src/chd_ct/imagechd/preprocess.py`、`common.py`、`geometry.py` | 先校验图像与标签网格，转接近 RAS 的规范方向；每例按 1/99 百分位裁剪并缩放到 0–1，可选统一数组大小；标签用离散插值。 |
| 病理切片颜色与 patch | `local-only/projects/radiopath/tiatoolbox/tools/stainnorm.py`、`stainextract.py`、`tissuemask.py`、`patchextraction.py` | Macenko/Vahadane 等染色分离或归一化、组织掩膜和多分辨率 patch 采样，适用于 WSI，不用于 CT/MR 的 HU 或空间坐标处理。 |
| 对照通用框架原理 | `upstream/nnunet/nnunetv2/preprocessing/`、`upstream/monai/monai/transforms/`、`upstream/torchio/src/torchio/transforms/`、`upstream/simpleitk/` | 官方核心源码由 `upstream/sources.json` 固定仓库、commit、许可和逐文件哈希；这里的项目摘录与上游源码分开保存。 |

## 预处理原理对照

从 DICOM 到训练张量可分为四层。第一层是**读取与物理解释**：pydicom 负责像素解码、LUT/窗位等 DICOM 语义；AAOCA CT 转换器在此基础上应用 HU 缩放并建立物理仿射。第二层是**几何核对**：whole-heart QC 与 CHD 管线检查 shape、affine、spacing 和图像/标签对齐；重采样时图像可用连续插值，离散标签必须采用保持类别的插值。第三层是**强度与裁剪**：CHD 管线采用逐例 1/99 百分位，适合其固定复现协议；nnU-Net 则根据训练集指纹和通道类型决定裁剪、目标间距及归一化，不能把逐例归一化当成通用 CT 方案。第四层是**数据划分与溯源**：whole-heart 管线限制训练来源，CHD 管线支持明确的患者级划分；来源统计不得用验证或目标病例拟合。

WSI 另有独立链条：染色分离/归一化、组织掩膜、按分辨率切 patch。它处理的是光学染色与超大二维图像；不能直接移植到三维 CT/MR。Forge 的通用图像预处理器也不处理医学图像的物理空间或标签一致性，因此仅作非医学对照。

## 来源、许可与边界

下表的 commit 是本地快照或上游索引中**实际留有证据**的值；“未记录”并不表示当前在线分支的版本。已核对公开仓库 HEAD `714a0a5b9ec5fdea605cebe52c019844dbfa7855` 含 `AAOCA/`、`whole_heart/`、`chd-ct-reproduction/` 三个目录；这里不声称逐文件内容与本地选摘相同。`NOASSERTION` 表示没有找到可覆盖这份本地拷贝的许可凭据，不推断为开源授权。`local-code/` 是用户自有项目的精选代码归集；此举不另授第三方复用许可证。

| 项目 | 原始来源与源码入口 | 版本/commit 证据 | 许可凭据与导出位置 |
| --- | --- | --- | --- |
| whole-heart-pipeline | `code_project/whole_heart/project1/05_src/data_pipeline.py`、`prepare_nnunet.py` | 原目录无独立 Git 根；[公开仓库 `whole_heart/` 目录](https://github.com/tianwuasd/project/tree/714a0a5b9ec5fdea605cebe52c019844dbfa7855/whole_heart) 已核对 | 自有代码，未见许可证，`NOASSERTION`；`projects/local-code/whole-heart-pipeline/`。 |
| whole-heart-qc | `code_project/whole_heart/数据探索工具/medical.py` | 未记录 | 自有代码，未见许可证，`NOASSERTION`；`projects/local-code/whole-heart-qc/`。 |
| aaoca | `code_project/heart_disease/dicom_conversion/convert_ct.py`、`convert_us.py` | 未记录；此来源无独立 Git 根 | 自有代码，未见许可证，`NOASSERTION`；`projects/local-code/aaoca/`。 |
| aaoca-public-copy | `code_project/heart_disease/AAOCA_publish/AAOCA/convert_ct.py` 等；[对应仓库目录](https://github.com/tianwuasd/project/tree/714a0a5b9ec5fdea605cebe52c019844dbfa7855/AAOCA) | 本地仓库远端为 `tianwuasd/project.git`，本地 HEAD `7bf87ac21d5c749aefe118d7dcc55d17bf859920`；公开仓库目录见上述较新的 HEAD | 未见许可证；仅索引，未重复复制。 |
| chd-ct | `code_project/chd-ct-reproduction/src/chd_ct/imagechd/preprocess.py`；[公开仓库 `chd-ct-reproduction/` 目录](https://github.com/tianwuasd/project/tree/714a0a5b9ec5fdea605cebe52c019844dbfa7855/chd-ct-reproduction) | 本地仓库 HEAD `85bb3d081f28cd35ecbfd5670a5e7356ca7e3ce8`，未记录远端；公开仓库目录另经核对 | 自有代码，未见许可证，`NOASSERTION`；`projects/local-code/chd-ct/`。 |
| radiopath | `code_project/RadioPath/tiatoolbox/tools/stainnorm.py` 等；本地仓库远端 `shangqigao/RadioPath.git` | 本地 HEAD `4f5f320e3d026067fffc49dde31526bd30ed5318` | 根目录及嵌套 `tiatoolbox` 未见可适用许可文件，`NOASSERTION`；`local-only/projects/radiopath/`。不把同名上游 tiatoolbox 的许可自动套用给此拷贝。 |
| pydicom | `code_project/heart_disease/AAOCA_quality_review_20260919/deps/pydicom/pixels/processing.py` | 随包 `pydicom-3.0.2.dist-info` 记录版本 3.0.2；无源码 commit | 随包 `licenses/LICENSE`：主体 MIT，私有字典等部分另列 BSD 条款；许可随 3 个源码文件保留在 `projects/vendor/pydicom/`。 |
| dicom-audit-tools | `code_project/academic_tools/skills/scientific-agent-skills/skills/pydicom/scripts/` | 本地仓库远端 `K-Dense-AI/scientific-agent-skills.git`，HEAD `36d8f13a1e754618794bf42f417884940077b4ae` | 仓库根 `LICENSE.md` 为 MIT，随 7 个脚本保留在 `projects/vendor/dicom-audit-tools/`。 |
| forge-general-image-tools | `code_project/picture_train/stable-diffusion-webui-forge/modules_forge/supported_preprocessor.py` | 本地 HEAD `dfdcbab685e57677014f05a3309b48cc87383167` | 原 `LICENSE.txt` 为 AGPL-3.0；仅在 `local-only/projects/forge-general-image-tools/` 保存，对照用途。 |
| monai-tutorials | `code_project/whole_heart/参考资料/04_MONAI_逐步实操/` 两个 notebook | 下载笔记本未带 commit；官方核心源码另见 `upstream/sources.json` | 原地未见可核对的许可文件；清除 notebook 输出和元数据后存于 `local-only/projects/monai-tutorials/`。 |
| nnunet-docs | `code_project/whole_heart/参考资料/03_nnUNet_自动化基线/` 五份说明 | 下载说明未带 commit；官方核心源码另见 `upstream/sources.json` | 原地未见可核对的许可文件；`local-only/projects/nnunet-docs/`。 |

`upstream/` 中 nnU-Net、MONAI、TorchIO 和 SimpleITK 的官方**选定核心文件**及许可证是独立补充，并非本地项目的完整源码签出。nnU-Net/MONAI 在原本地项目中只有教程、文档和 API 调用；SimpleITK、TorchIO、Nibabel、dicom2nifti 也未发现完整本地源码项目。需要运行时应另用正式发行包和匹配的版本，不从这些摘录拼装环境。

本次项目清单共 11 项、选摘 45 个文件，其中 Python 源码 35 个；`local-code/` 17 个、`vendor/` 12 个、`local-only/projects/` 16 个。43 个文件与原件逐字节一致，另 2 个 MONAI notebook 清除了输出和元数据；未导出的例外为 0。许可分类按项目计：2 项有本地宽松许可依据，1 项为 AGPL-3.0，8 项为 `NOASSERTION`（其中一项仅索引、未复制）。

## 核验方法与限制

`local-only/projects/curate_local_projects.py` 是当时的一次性允许清单脚本，会遍历原始代码工作区；已移至本地目录。公开 JSON 使用逻辑路径，本地原始路径只在 `local-only/projects/source-*.json` 保存。`local-only/projects/verify_exports.py` 是只读核验器：按清单重算每个原件和导出文件的 SHA-256、检查导出 Python 语法，并检查公开索引是否出现本机盘符、凭据形式或误入的影像/病例文件。它不会导入或运行摘录源码。notebook 仅清除了输出、执行次数、元数据、单元格 ID 与附件，故其导出哈希与原件不同；其余文件要求字节一致。

许可核验只确认本地附带文件和所列快照，不代表审查了所有依赖或线上最新版。隐私核验限于允许清单、文件类型和静态文本检查；DICOM 去标识脚本与 AAOCA 转换器不能代替针对真实数据的像素烧录信息、自由文本和外部记录人工审核。
