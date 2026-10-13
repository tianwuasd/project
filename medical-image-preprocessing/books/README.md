# 医学图像预处理：本地书籍导读

这里收录三本本地 PDF 的原创中文导读和[预处理专题笔记](preprocessing-topics.md)。每个判断都尽量标出章节、PDF 物理页和书中印刷页；机器可读的核对记录见 [book_sources.json](book_sources.json)。PDF 原文和定向提取文本只留在 `local-only/books/`，本目录只放归纳与定位索引。

| 书 | 适合解决的问题 | 导读 |
| --- | --- | --- |
| Geoff Dougherty, *Digital Image Processing for Medical Applications*（2009） | 灰度/直方图、几何变换与插值、噪声、形态学和传统分割的基础 | [阅读导引](dougherty-digital-image-processing.md) |
| R. Indrakumari 等编, *Deep Learning in Medical Image Analysis: Recent Advances and Future Trends*（2025） | 眼底图像标准化、皮肤图像预处理以及深度学习分割的任务背景 | [阅读导引](recent-advances-medical-image-analysis.md) |
| Archana Mire 等编, *Advances in Deep Learning for Medical Image Analysis*（2022） | 胸片、血涂片、皮肤镜、糖尿病足图像的具体预处理案例 | [阅读导引](advances-deep-learning-medical-image-analysis.md) |

**页码规则。** “PDF 页”从文件第一页起计，1-based；“印刷页”是书页上可见的阿拉伯数字。三书正文各有前置页，所以两者不能混用。索引给出的是实际已提取的范围，笔记只引用其中核对过的内容；它们不是逐页通读全书的声明。含图、公式的 PDF 转文字有错码，关键结论以能辨认的标题和叙述为据，不从残缺公式恢复参数。

**使用顺序。** 先用 Dougherty 建立“改变像素值”和“改变空间坐标”的区别，再看[专题笔记](preprocessing-topics.md)中七个环节的适用条件与检查点。两本论文集用于观察特定图像任务中的流程，不宜把其输入尺寸、阈值或归一化参数直接当作 CT/MRI 通用默认值。nnU-Net、MONAI、SimpleITK 的对应关系是工程对照建议，不表示原书使用了这些软件。

**代码状态。** 已读页中，Dougherty 有公式、算例及 ImageJ 操作练习；两本论文集有模型/预处理流程描述，2022 年文集第 2 章还给出软件栈与训练参数。没有在已检查的相关章节发现可直接运行的完整源代码或明确标为伪代码的算法块；书的封底/简介对附带脚本的宣传不能替代正文核验。未读页不能据此下结论。
