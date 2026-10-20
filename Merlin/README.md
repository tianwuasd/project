# Merlin 中文学习入口

Merlin 是面向三维 CT 的视觉语言基础模型：用 CT、放射科报告和诊断编码共同学习表征，再用于图文检索、分类、疾病预测、报告生成及分割。这个目录收录官方源码快照、论文笔记和按代码调用顺序编排的中文指南，方便从“知道论文做了什么”走到“能找到对应实现”。

**第一次看，从 [入门与阅读顺序](docs/03-getting-started.md) 开始。** 第一轮目标是弄清一个 CT 如何变成 2048 维特征，再决定是否做下游实验。

## 阅读材料

| 材料 | 解决的问题 |
| --- | --- |
| [项目介绍](docs/01-introduction.md) | Merlin 是什么，哪些任务和代码已公开 |
| [源码阅读笔记](docs/02-code-reading.md) | 预处理、模型构造、前向传播和输出如何对应 |
| [入门与阅读顺序](docs/03-getting-started.md) | 从哪几个文件看起，如何准备环境和运行第一例 |
| [先心 CT 迁移思考](docs/04-chd-transfer-notes.md) | 哪些思路值得借鉴，最小对照实验怎样安排 |
| [论文精读笔记](docs/05-paper-notes.md) | 按本地 arXiv v2 正文和附录整理的方法、结果与局限 |
| [来源与版本](UPSTREAM.md) | 上游地址、固定提交、许可证和更新方式 |

## 目录

```text
Merlin/
├── README.md
├── UPSTREAM.md
├── UPSTREAM_MANIFEST.json       # 上游逐文件 SHA-256 清单
├── docs/                       # 新增中文介绍与学习笔记
└── upstream/                   # 官方仓库的完整受版本控制文件快照
    ├── README.md
    ├── LICENSE
    ├── merlin/                 # 模型与数据处理
    ├── documentation/          # 示例、说明及特征提取入口
    └── pyproject.toml
```

`upstream/` 固定为官方提交 `e1552df771367aa2589ec7dcfd3151dc91f8ce74`，未修改其中的文件。这里只保存受版本控制的代码和文档，没有下载模型权重或 CT 数据。独立的 Merlin-nnUNet 仓库在指南中提供入口，未混入这份主仓库快照。

## 先记住三个区别

1. **2048 维图像特征**是图像骨干池化后的表示；**512 维图文特征**是对比学习空间中的表示，两者用途不同。
2. `Merlin(ImageEmbedding=True)` 表示前向传播只需要图像；当前构造函数仍会创建文本编码器，因此第一次初始化仍可能下载 Clinical Longformer。
3. 当前公开代码能够加载权重和执行推理，但不包含完整的 CT／报告／EHR 联合预训练训练器。报告生成模块虽然有损失计算接口，其外层 `forward` 处于 `no_grad` 中，不能直接当作训练循环。

上述行为对应本目录的固定版本，细节和源码位置见[源码阅读笔记](docs/02-code-reading.md)。

## 原始资源

- [官方 Merlin 仓库](https://github.com/StanfordMIMI/Merlin)
- [Nature 论文](https://doi.org/10.1038/s41586-026-10181-8)
- [arXiv 论文](https://arxiv.org/abs/2406.06512)
- [预训练权重](https://huggingface.co/stanfordmimi/Merlin)
- [Merlin-nnUNet 分割仓库](https://github.com/ashwinkumargb/Merlin-nnUNet)

上游代码采用 [MIT 许可证](upstream/LICENSE)，原作者声明完整保留。中文材料是学习笔记，不属于原作者官方文档。本次整理完成源码与文档的静态核对，尚未安装训练环境或运行模型推理。
