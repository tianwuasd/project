# 代码总索引

以下均为汇集的源码或教程；是否需要整包依赖、是否清除输出，以对应目录 README 和来源清单为准。本次未执行这些项目或 notebook。

## 用户项目预处理代码

- [aaoca/common.py](projects/local-code/aaoca/common.py)
- [aaoca/convert_ct.py](projects/local-code/aaoca/convert_ct.py)
- [aaoca/convert_us.py](projects/local-code/aaoca/convert_us.py)
- [aaoca/patient_pipeline/imaging.py](projects/local-code/aaoca/patient_pipeline/imaging.py)
- [chd-ct/src/chd_ct/imagechd/augment.py](projects/local-code/chd-ct/src/chd_ct/imagechd/augment.py)
- [chd-ct/src/chd_ct/imagechd/common.py](projects/local-code/chd-ct/src/chd_ct/imagechd/common.py)
- [chd-ct/src/chd_ct/imagechd/geometry.py](projects/local-code/chd-ct/src/chd_ct/imagechd/geometry.py)
- [chd-ct/src/chd_ct/imagechd/preprocess.py](projects/local-code/chd-ct/src/chd_ct/imagechd/preprocess.py)
- [chd-ct/src/chd_ct/recovery.py](projects/local-code/chd-ct/src/chd_ct/recovery.py)
- [whole-heart-pipeline/05_src/appearance.py](projects/local-code/whole-heart-pipeline/05_src/appearance.py)
- [whole-heart-pipeline/05_src/data_pipeline.py](projects/local-code/whole-heart-pipeline/05_src/data_pipeline.py)
- [whole-heart-pipeline/05_src/prepare_nnunet.py](projects/local-code/whole-heart-pipeline/05_src/prepare_nnunet.py)
- [whole-heart-pipeline/tests/test_appearance.py](projects/local-code/whole-heart-pipeline/tests/test_appearance.py)
- [whole-heart-pipeline/tests/test_data_pipeline.py](projects/local-code/whole-heart-pipeline/tests/test_data_pipeline.py)
- [whole-heart-pipeline/tests/test_preparation_markers.py](projects/local-code/whole-heart-pipeline/tests/test_preparation_markers.py)
- [whole-heart-qc/medical.py](projects/local-code/whole-heart-qc/medical.py)
- [whole-heart-qc/tests/test_medical.py](projects/local-code/whole-heart-qc/tests/test_medical.py)

## 本地找到的开源工具源码

- [dicom-audit-tools/skills/pydicom/scripts/_common.py](projects/vendor/dicom-audit-tools/skills/pydicom/scripts/_common.py)
- [dicom-audit-tools/skills/pydicom/scripts/anonymize_dicom.py](projects/vendor/dicom-audit-tools/skills/pydicom/scripts/anonymize_dicom.py)
- [dicom-audit-tools/skills/pydicom/scripts/deidentification_audit.py](projects/vendor/dicom-audit-tools/skills/pydicom/scripts/deidentification_audit.py)
- [dicom-audit-tools/skills/pydicom/scripts/dicom_inventory.py](projects/vendor/dicom-audit-tools/skills/pydicom/scripts/dicom_inventory.py)
- [dicom-audit-tools/skills/pydicom/scripts/pixel_frame_planner.py](projects/vendor/dicom-audit-tools/skills/pydicom/scripts/pixel_frame_planner.py)
- [dicom-audit-tools/skills/pydicom/scripts/transfer_syntax_inspector.py](projects/vendor/dicom-audit-tools/skills/pydicom/scripts/transfer_syntax_inspector.py)
- [dicom-audit-tools/skills/pydicom/scripts/uid_mapping_validator.py](projects/vendor/dicom-audit-tools/skills/pydicom/scripts/uid_mapping_validator.py)
- [pydicom/pydicom/pixel_data_handlers/util.py](projects/vendor/pydicom/pydicom/pixel_data_handlers/util.py)
- [pydicom/pydicom/pixels/processing.py](projects/vendor/pydicom/pydicom/pixels/processing.py)
- [pydicom/pydicom/pixels/utils.py](projects/vendor/pydicom/pydicom/pixels/utils.py)

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

课程中另外提取的 36 个片段保存在本地 `local-only/knowledge/`；[课程代码清单](knowledge/code-inventory.md)给出定位与注意事项。电子书已读相关章节未核验到完整可运行代码，详见[书籍索引](books/README.md)。
