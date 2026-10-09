# 主流程数据规范

输入为已解压 ImageCHD：`ct_编号_image.nii.gz` 与 `ct_编号_label.nii.gz`。也可传其上层目录，程序会识别 ImageCHD_dataset 子目录。不会隐式解压 archive。

训练标签：0背景；1LV；2RV；3LA；4RA；5MYO；6AO；7PA。额外非负整数标签变成255；负值/小数报错。任一目标结构缺少标注时整例排除，并记录原因。不能直接混用旧论文十结构编号。

原影像和标签必须同形状、同affine。预处理统一方向、按逐例百分位归一化，默认保持原始网格，不猜测HU或真实毫米间距。输出缓存可以整体移动；不依赖旧Windows原始路径。

显式划分CSV列为 `case_id,patient_id,split`，split只能是train/val/test；覆盖所有保留病例，患者不得跨集合。默认文件名患者编号是假设，正式研究应核实。

无标签新影像使用 `--for-prediction`，可接受单个NIfTI或标准命名目录。该缓存不生成训练划分，不能训练或评估。

更详细命令、旧v1缓存兼容规则与输出说明见 [使用流程](workflow.md)。原archive核查结果见 [检查报告](imagechd-archive.md)。
