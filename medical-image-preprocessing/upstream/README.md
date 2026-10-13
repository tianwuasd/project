# 官方预处理源码精选

本地未发现完整 nnU-Net/MONAI 源码树，因此在本地教程之外补充这些官方源码。采集日期为 2026-10-13；准确 commit、固定版本 URL、许可证与每个文件的 SHA-256 见 [sources.json](sources.json)。

| 项目 | 本次收录 | 建议阅读入口 |
| --- | --- | --- |
| nnU-Net | 裁剪、归一化映射、CT/Z-score 归一化、重采样、预处理器、数据指纹 | [default_preprocessor.py](nnunet/nnunetv2/preprocessing/preprocessors/default_preprocessor.py) |
| MONAI | spatial、intensity、croppad 的 array 与 dictionary 实现 | [空间字典变换](monai/monai/transforms/spatial/dictionary.py)、[强度字典变换](monai/monai/transforms/intensity/dictionary.py) |
| TorchIO | 归一化、标准化、直方图标准化、截断、重定向、空间变换、裁剪填充与参考空间 | [normalize.py](torchio/src/torchio/transforms/intensity/normalize.py)、[spatial.py](torchio/src/torchio/transforms/spatial/spatial.py) |
| SimpleITK | 变换与重采样、变换基础、分割概览、配准入门、数据增广，共 5 个 notebook | [变换与重采样](simpleitk/Python/21_Transforms_and_Resampling.ipynb)、[配准入门](simpleitk/Python/60_Registration_Introduction.ipynb) |

## 读代码时核对什么

**nnU-Net：** `CTNormalization` 使用数据指纹提供的分位点、均值和标准差，而 `ZScoreNormalization` 按影像自身统计，并可依照掩码确定统计区域。因此“都叫归一化”并不代表作用相同。进一步顺着预处理器检查重排轴、裁剪边界、目标 spacing、标签采样位置以及预测时还原所需的信息。

**MONAI：** `array.py` 处理单个对象，`dictionary.py` 通过键组织图像、标签及相关元数据。查看 `Spacingd`、`Orientationd`、`CropForegroundd` 等时，关注不同键的插值方式与空间操作是否一致；查看强度变换时，确认它只作用于影像键。

**TorchIO：** 本快照已包含 v1→v2 的接口迁移。老材料中的 `RescaleIntensity`/`ZNormalization` 与这里的 `Normalize`/`Standardize` 应按[迁移说明](torchio/docs/get-started/migration.md)核对；部分 `Random*` 名称也已调整。不要把不同年代教程中的参数直接混用。这里保留源码，不声明这些接口适用于本机已有安装版。

**SimpleITK：** 从索引坐标与物理坐标的关系读起，核对输出网格、变换方向、插值器和边界填充值。旧的 `SimpleITK/SimpleITK-Notebooks` 仓库首页已指向 `InsightSoftwareConsortium/SimpleITK-Notebooks`，本次从后者采集。

## 使用限制与改动

四个上游快照均带各自 `LICENSE`。Python 源文件按原始内容保存；notebook 移除了输出和附带元数据，并添加来源与改动说明。本目录不是完整安装包，未执行 notebook，也未取得其外部演示数据。某些上游 Markdown 内链依赖完整官方仓库，应回到 `sources.json` 中的上游位置阅读。
