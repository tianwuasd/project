# 代码总索引

全部项目代码来自第三方官方公开仓库。固定版本、许可证及哈希见 [来源清单](upstream/sources.json)。这些是阅读用源码精选，运行时需要完整项目及其依赖。

## nnU-Net 官方核心实现

- [nnunetv2/experiment_planning/dataset_fingerprint/fingerprint_extractor.py](upstream/nnunet/nnunetv2/experiment_planning/dataset_fingerprint/fingerprint_extractor.py)
- [nnunetv2/preprocessing/cropping/cropping.py](upstream/nnunet/nnunetv2/preprocessing/cropping/cropping.py)
- [nnunetv2/preprocessing/normalization/default_normalization_schemes.py](upstream/nnunet/nnunetv2/preprocessing/normalization/default_normalization_schemes.py)
- [nnunetv2/preprocessing/normalization/map_channel_name_to_normalization.py](upstream/nnunet/nnunetv2/preprocessing/normalization/map_channel_name_to_normalization.py)
- [nnunetv2/preprocessing/preprocessors/default_preprocessor.py](upstream/nnunet/nnunetv2/preprocessing/preprocessors/default_preprocessor.py)
- [nnunetv2/preprocessing/resampling/default_resampling.py](upstream/nnunet/nnunetv2/preprocessing/resampling/default_resampling.py)
- [nnunetv2/preprocessing/resampling/resample_torch.py](upstream/nnunet/nnunetv2/preprocessing/resampling/resample_torch.py)

## MONAI 官方变换实现

- [monai/transforms/croppad/array.py](upstream/monai/monai/transforms/croppad/array.py)
- [monai/transforms/croppad/dictionary.py](upstream/monai/monai/transforms/croppad/dictionary.py)
- [monai/transforms/intensity/array.py](upstream/monai/monai/transforms/intensity/array.py)
- [monai/transforms/intensity/dictionary.py](upstream/monai/monai/transforms/intensity/dictionary.py)
- [monai/transforms/spatial/array.py](upstream/monai/monai/transforms/spatial/array.py)
- [monai/transforms/spatial/dictionary.py](upstream/monai/monai/transforms/spatial/dictionary.py)

## TorchIO 官方变换实现

- [src/torchio/transforms/intensity/clamp.py](upstream/torchio/src/torchio/transforms/intensity/clamp.py)
- [src/torchio/transforms/intensity/histogram_standardization.py](upstream/torchio/src/torchio/transforms/intensity/histogram_standardization.py)
- [src/torchio/transforms/intensity/normalize.py](upstream/torchio/src/torchio/transforms/intensity/normalize.py)
- [src/torchio/transforms/intensity/standardize.py](upstream/torchio/src/torchio/transforms/intensity/standardize.py)
- [src/torchio/transforms/spatial/crop_or_pad.py](upstream/torchio/src/torchio/transforms/spatial/crop_or_pad.py)
- [src/torchio/transforms/spatial/reorient.py](upstream/torchio/src/torchio/transforms/spatial/reorient.py)
- [src/torchio/transforms/spatial/spatial.py](upstream/torchio/src/torchio/transforms/spatial/spatial.py)
- [src/torchio/transforms/spatial/to_reference_space.py](upstream/torchio/src/torchio/transforms/spatial/to_reference_space.py)

## SimpleITK 官方教程

- [Python/21_Transforms_and_Resampling.ipynb](upstream/simpleitk/Python/21_Transforms_and_Resampling.ipynb)
- [Python/22_Transforms.ipynb](upstream/simpleitk/Python/22_Transforms.ipynb)
- [Python/300_Segmentation_Overview.ipynb](upstream/simpleitk/Python/300_Segmentation_Overview.ipynb)
- [Python/60_Registration_Introduction.ipynb](upstream/simpleitk/Python/60_Registration_Introduction.ipynb)
- [Python/70_Data_Augmentation.ipynb](upstream/simpleitk/Python/70_Data_Augmentation.ipynb)

## pydicom 官方像素处理

- [src/pydicom/pixel_data_handlers/util.py](upstream/pydicom/src/pydicom/pixel_data_handlers/util.py)
- [src/pydicom/pixels/processing.py](upstream/pydicom/src/pydicom/pixels/processing.py)
- [src/pydicom/pixels/utils.py](upstream/pydicom/src/pydicom/pixels/utils.py)

课程片段仅在本地保存，见[课程代码清单](knowledge/code-inventory.md)。电子书相关章节的代码核对情况见[书籍索引](books/README.md)。
