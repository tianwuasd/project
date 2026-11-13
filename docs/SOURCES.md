# 模型及数值来源

- STU-Net-S：uni-medical/STU-Net，Apache-2.0，上游结构文件 blob `85587aca5fdf1180dc266a138f3ae97c231875b2`。本项目调整类与接口组织、固定 small 尺寸、配置输出类别和拓扑，保留对应参数名称与结构。许可全文见 STUNET_LICENSE.txt。
- 官方 STU-Net-S `small_ep4k.model`：SHA256 `f440f401bf4cac1d1d6f7c4635542ef057193a5f96c0b3c661a6179afeab7be6`。仅加载骨干，独立初始化数据集输出头；权重不随代码分发。
- CHD1282 边际损失：本次计划固定来源 `tianwuasd/project@8ea9736c14aa230388483f204065bad4fa46797d` 的 dataset1282_analysis；逐病例未知集合、合并背景组及包含该组的 Dice。
- ImageCHD 部分监督：本次核查 chd-ct-panel-v2 的 PartialDiceCE；缺 MYO 的背景允许 0/5，CE 按有效体素归约，前景 Dice 排除该病例未标类别。
- 确定性预处理：直接调用 nnU-Net 2.8.1 的 DefaultPreprocessor、PlansManager 和重采样函数；仅将前景坐标收集改为有界抽样。MONAI 1.5.2 组织 Dataset/Compose 和滑窗推断。

患者数据、Excel、人工复核政策、原始划分和运行回执保留在私有工作目录，不随代码分发。8/11 类数据空间和损失配方分别保留。
