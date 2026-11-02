# 输入与输出合同

数据根包含 `manifest.json`（记录列表）。每条记录至少有 `case_id`、`source_key`、`collection`；collection只接受native与review_required。case_id为安全的字母、数字、下划线或连字符，且不能重复。

规范文件路径为 `collection/images/CASE_0000.nii.gz` 和 `collection/labels/CASE.nii.gz`。程序从数据根重建这些路径，可迁移原Windows清单到Linux；拒绝越界和链接路径。extracted_labels、历史QC等目录不作为第二批样本。outputs中的原始sha256按文件名匹配，用于本轮完整读取后比较。

Excel工作表默认dicom2nii_info，字段顺序如下：

```text
idx, PatientName, PatientID, PatientBirthDate, PatientSex,
AcquisitionDate, ManufacturerModelName, PixelSpacingX, PixelSpacingY,
SliceThickness, Rows, Cols, all_file_z, all_read_file_z, valid_z,
calc_zSpacing, dicom_folder
```

来源字段保留在私有输出。日期规范为ISO文本；PatientID不用于模糊关联，数值类型ID会标记可能丢失精度。键不唯一时不强行匹配；每条Excel行都有严格匹配、未匹配或关联歧义去向。未匹配行不会从数据总体中静默消失。

配置优先级为CLI显式参数 > `--config`任务配置 > `project.local.json` > 当前release默认值。配置文件中的相对路径相对该配置所在目录解释；未知字段拒绝。work_root与environment不能位于源码或数据目录内部，也不能包含数据根。

`cases/CASE.json`保存科学指标与错误；对应`.complete.json`提交结果摘要和合同。只有完整标记、合同、结果SHA与两份实际输入SHA均匹配的成功病例可恢复复用。读取失败、只读头结果和不完整文件不冒充全量体素完成。

正式报告中的分母包括清单文件对、终态病例、成功分析、可用图像统计、可用标签统计、Excel记录及严格匹配记录。多个复核事项可以属于同一病例，不能把问题条数相加当作异常病例数。

数据、环境、报告、代码各自独立。公开测试用运行时生成的合成影像与Excel，不含真实个体字段；CSV防止公式开头文本被Excel作为公式执行，JSON保留原字符串。
