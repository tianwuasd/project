# STU-Net-S预处理设计与实现索引

当前已实现 `--method stunet-s`；运行和合同见[使用说明](preprocessing-usage.md)。以下保留方法依据，执行范围以各次输出summary为准。

参考一手资料后的建议主线：
1. 原始数据只读，标签及QC合同明确后生成衍生副本。
2. 校核物理间距来源和数组轴，训练子集拟合规划与统计。
3. 影像非零裁剪，保存可逆元数据。
4. 以nnU-Net CTNormalization建立基线，显式配置normalizer；逐例Z-score作为对照。
5. 按锁定nnU-Net实现先归一化再重采样，验证离散标签、保留结构与输出可恢复性。
6. 有界CPU/RAM/读取并发、逐例原子完成、同合同恢复。

数据集私有问题、筛查结果与选择建议保存在outputs/research中，不随代码上传。CTNormalization是当前预处理配置；逐例Z-score尚未注册，不声称已完成效果对比。

一手来源：
- [ImageCHD作者标签说明](https://github.com/XiaoweiXu/ImageCHD-A-3D-Computed-Tomography-Image-Dataset-for-Classification-of-Congenital-Heart-Disease)
- [nnU-Net归一化](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation_normalization.md)
- [nnU-Net原生预处理](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/preprocessing/preprocessors/default_preprocessor.py)
- [STU-Net](https://github.com/uni-medical/STU-Net)

ImageCHD作者说明额外编号可忽略；这不等于指定某一种背景/ignore损失映射。相关PCL作者代码采用过额外编号归背景及逐例百分位+Z-score，但其CHD68数据与2D流程不同，不能作为当前工程直接复制的模板。
已适配：本机锁定2.8.1原始前景采样使用正标签条件，项目显式限定目标1..7，通过合成ignore样本及fit_ids验证，不修改全局包。CT配置作为当前基线，效果需后续同协议训练比较。
