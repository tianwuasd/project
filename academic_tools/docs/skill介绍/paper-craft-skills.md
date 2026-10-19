# Paper Craft Skills

> 版本说明（2026-10-19）：源码已更新；正文保留此前整理的指南，功能与安装入口以当前上游 README 为准。见[本次更新记录](../../UPDATE-2026-10-19.md)。

- **上游仓库**：[zsyggg/paper-craft-skills](https://github.com/zsyggg/paper-craft-skills)
- **本地目录**：[`skills/paper-craft-skills`](../../skills/paper-craft-skills/)
- **项目类型**：论文深度解读、方法图解、视觉文章和学术幻灯片 skill 集
- **文章评价**：第一篇文章介绍了该项目，但未给出分档；第二篇文章未重点测评

## 一句话定位

Paper Craft Skills 把论文或 arXiv 链接转为更容易阅读和汇报的视觉材料：方法图解、深度长文、Mermaid 架构图、公式/代码对照以及 PPTX/PDF 幻灯片。

## 核心功能

- **paper-comic**：先读论文并推荐图解范围，再按 `paper-figure` 或 `sketchnote` 风格生成方法图、机制图和关键结果图。
- **paper-analyzer**：阅读论文全文、查找 GitHub 源码，用 storytelling、academic 或 concise 风格写成带公式、表格、代码对照和 Mermaid 图的 HTML 长文。
- **paper-deck**：从论文/文章/技术笔记生成 deck brief、逐页大纲、视觉提示词、16:9 slide image，并导出 PPTX/PDF。
- **真实素材利用**：可以规划 PDF 中的论文图、表格、实验曲线或截图如何裁切并放入页面，而不是完全依赖生成图像。
- **逐页返修**：每页 deck 有独立 prompt，便于只重生成某一页或保留布局更换视觉内容。

## 适用场景

适合组会汇报、论文预读、课程讲解、方法图解、技术博客、海报和面向非本领域读者的论文传播。它重点处理视觉沟通，不替代严谨的文献综述、实验统计或最终论文写作。

## 优势

1. 从论文内容出发规划视觉叙事，减少先做模板再硬塞内容的问题。
2. 同时覆盖图解、长文和幻灯片，输出形态丰富。
3. 提供 academic、concise、journal-minimal 等面向研究沟通的风格。
4. 允许使用真实论文素材，并支持逐页返修，适合迭代式制作。

## 局限与风险

- “像专家手笔”或“高质感”是项目定位，不等于图中的数字、结构和结论已经被核验。
- 生成式图像可能错画模型结构、箭头、公式、单位和实验结果；论文级图片必须由作者逐项校对。
- deck 和 HTML 依赖浏览器、公式渲染、图片生成和 PPTX/PDF 导出环境，零配置说法不代表所有环境都无需依赖。
- 不应把视觉美化当成论文贡献或实验可信度的证明。

## 文章评价

第一篇文章把它定位为“出图制图方便”的论文视觉化工具，适合将复杂论文转成 slides、poster 或社交媒体材料；文章没有给出分档。该评价适合用于判断方向，不替代对具体图件的事实核查。

## 推荐组合

- 用 Claude Scholar 或 Scientific Agent Skills 先整理论文证据和结果，再用 Paper Craft 做可视化。
- 用 PaperSpine 先确定贡献和 Results 映射，再用 paper-comic 或 paper-deck 讲清核心方法。
- 用 Nature Skills 做英文学术图注和论文风格润色，保持视觉与正文术语一致。

## 使用前核查

1. 生成前提供论文全文、真实图表、公式和来源，避免让 Agent 猜结构。
2. 对每个箭头、变量、单位、数据点、图例和引用做逐项校对。
3. 保留源文件、生成 prompt、真实素材和导出版本，确保可返修。
4. 发布前检查字体、版权、图片许可和 PPTX/PDF 跨设备显示。

## 参考

- [仓库中文 README](../../skills/paper-craft-skills/README.zh.md)
- [知乎文章一](https://www.zhihu.com/question/2058648818057531853/answer/2065455008460624405)
- [知乎文章二](https://www.zhihu.com/question/2058648818057531853/answer/2063668103637210722)
