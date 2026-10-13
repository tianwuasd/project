# 第三方公开项目导读

本资料集的项目来源限定为五个官方公开项目。选择依据是官方仓库可核对、预处理用途明确、版本与许可可追溯；这不等于保证任意参数在所有任务上有效。

| 项目与官方来源 | 项目定位 | 重点流程与源码入口 |
| --- | --- | --- |
| [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) | 根据数据特征规划的医学图像分割框架 | 数据指纹 → 规划 → 重排轴 → 非零区域裁剪 → 归一化 → 重采样。读 [预处理器](../upstream/nnunet/nnunetv2/preprocessing/preprocessors/default_preprocessor.py)与[归一化](../upstream/nnunet/nnunetv2/preprocessing/normalization/default_normalization_schemes.py)。 |
| [MONAI](https://github.com/Project-MONAI/MONAI) | 可组合的医学影像深度学习工具集 | 显式配置读取、方向、间距、强度、裁剪及训练增广。读[空间字典变换](../upstream/monai/monai/transforms/spatial/dictionary.py)、[强度字典变换](../upstream/monai/monai/transforms/intensity/dictionary.py)、[裁剪字典变换](../upstream/monai/monai/transforms/croppad/dictionary.py)。 |
| [TorchIO](https://github.com/TorchIO-project/torchio) | 组织病例影像、标签与空间/强度变换 | 围绕病例组合方向整理、重采样、裁剪填充和标准化。读[方向](../upstream/torchio/src/torchio/transforms/spatial/reorient.py)、[标准化](../upstream/torchio/src/torchio/transforms/intensity/standardize.py)、[裁剪填充](../upstream/torchio/src/torchio/transforms/spatial/crop_or_pad.py)。 |
| [SimpleITK 官方教程](https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks) | 医学图像物理空间、变换、配准与传统处理 | 定义参考网格与坐标映射 → 选择插值器 → 重采样 → 核对输出几何。读[变换与重采样](../upstream/simpleitk/Python/21_Transforms_and_Resampling.ipynb)和[配准入门](../upstream/simpleitk/Python/60_Registration_Introduction.ipynb)。 |
| [pydicom](https://github.com/pydicom/pydicom) | DICOM 读取、像素解码和强度解释基础库 | 区分存储像素、modality LUT/rescale、VOI LUT/显示窗口及颜色处理。读[像素处理](../upstream/pydicom/src/pydicom/pixels/processing.py)和[像素工具](../upstream/pydicom/src/pydicom/pixels/utils.py)。 |

## 如何比较流程

nnU-Net 提供数据驱动的整体规划；MONAI 与 TorchIO 提供由使用者组合的组件；SimpleITK 与 pydicom 提供几何处理和底层读取能力。后两者不是自动完成所有训练准备的端到端分割方案。

nnU-Net 的 CT 标准化使用训练数据前景汇总的分位点、均值和标准差；MONAI 的具体范围由变换参数确定；TorchIO 的 Standardize 可以从选定掩码内计算统计量。相同的“归一化”名称并不保证统计范围或输出意义相同。

空间处理先明确方向、spacing、参考网格与插值规则。图像与标签应同步几何变换，强度处理只作用于影像；统一数组尺寸与统一物理间距是不同操作。裁剪与重采样的参数应记录，以便将预测结果放回原空间。

## 版本与使用范围

完整 commit、来源 URL、许可和逐文件哈希见 [inventory.json](inventory.json)、[file-manifest.json](file-manifest.json)和[详细来源](../upstream/sources.json)。pydicom 固定为官方 v3.0.2；其余项目固定到已记录的 commit。TorchIO 快照含 v2 接口，旧教程应对照[迁移说明](../upstream/torchio/docs/get-started/migration.md)。

所有源文件都是学习用精选，未复制完整依赖。Python 源码保留官方字节；SimpleITK notebook 清除了输出及无关元数据。学习笔记说明来源与工程推论，不作为官方新增规范。
