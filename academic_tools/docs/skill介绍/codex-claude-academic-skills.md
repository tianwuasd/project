# Codex Claude Academic Skills

> 版本说明（2026-10-19）：源码已更新；正文保留此前整理的指南，功能与安装入口以当前上游 README 为准。见[本次更新记录](../../UPDATE-2026-10-19.md)。

- **上游仓库**：[zLanqing/codex-claude-academic-skills](https://github.com/zLanqing/codex-claude-academic-skills)
- **本地目录**：[`skills/codex-claude-academic-skills`](../../skills/codex-claude-academic-skills/)
- **项目类型**：面向中国科研工作者的中文优先三技能包
- **文章评价**：第二篇文章评为“人上人”

## 一句话定位

该仓库把论文写作、学术 Office 文档和科研计算组合在一起，重点覆盖 MATLAB/Python 仿真、信号处理、统计分析、论文配图、可编辑 Word/PPT 以及中文优先的论文写作。

## 核心功能

- **research-writing-skill**：论文摘要、引言、方法、实验、结论、修改、润色、论证规划和审稿回复。
- **office-academic-skill**：从 PDF/论文生成文献阅读报告、结构化 DOCX、组会/开题/答辩 PPT，并支持已有 Office 文档编辑。
- **scientific-toolkit-skill**：MATLAB/Octave 信号与图像处理、FFT、滤波、矩阵计算、仿真、论文配图导出，以及 NumPy、SciPy、pandas、matplotlib、scikit-learn、statsmodels、SymPy、pymoo、QuTiP、pymatgen、TimesFM 等 Python 工具。
- **文献与引用**：README 列出 arXiv、PubMed、Crossref、Semantic Scholar、OpenAlex 等查找方向，并支持 DOI 到 BibTeX、元数据提取和引文验证。
- **中文规范**：默认中文，保留英文标题、公式、变量、方法名、命令和参考文献；要求不编造 DOI、实验值和图表结论。

## 适用场景

适合光电信息、信号处理、传感器、光学、工程仿真等方向的科研者，也适合需要快速做学术 Word/PPT 的中文研究团队。它是领域明确的个人工具箱，不是覆盖所有学科的统一研究平台。

## 优势

1. MATLAB/信号处理细节较通用助手更有针对性，适合工程研究。
2. 写作、科研计算和 Office 产物可以在一个仓库内衔接。
3. 中文优先并明确列出“不编造数据”和来源标注规范。
4. 结构简单，三个独立 skill 可按需安装到 Claude Code 或 Codex。

## 局限与风险

- 领域重点偏光电信息与工程，医学、社科和湿实验需求需要其他 skill 补充。
- README 中的库列表提供指导，不代表所有依赖已安装或所有代码都能在当前平台直接运行。
- MATLAB/Python 代码、FFT 窗函数、频率分辨率、统计模型和论文图都必须用真实数据复核。
- Office 生成只能保证结构性辅助，不能自动保证模板、公式、字体和投稿规范完全正确。

## 文章评价

第二篇文章评为“人上人”，特别称赞其 MATLAB 信号处理会考虑窗函数和频率分辨率等工程细节，同时指出它的质控不像严谨学术流水线，更像作者的个人工具箱。该评价与仓库的领域定位相符，但具体模块仍要按自己的数据测试。

## 推荐组合

- 用 scientific-toolkit-skill 生成真实实验结果和图表，再用 research-writing-skill 写方法/实验章节。
- 用 office-academic-skill 做组会或答辩 PPT，再用 Paper Craft 做方法图和视觉优化。
- 用 ARS-Codex、Claude Scholar 或 Nature Skills 补充文献证据与英文表达。

## 使用前核查

1. 先确认 MATLAB/Octave、Python 库、PowerPoint/Word 等依赖和版本。
2. 对 FFT、滤波、统计检验、模型指标和图注逐项核验，不让 Agent 猜参数。
3. 检查 DOCX/PPTX 是否可编辑、是否保留来源、单位、公式和版权信息。
4. 运行代码前阅读相关 `SKILL.md`、脚本和引用说明。

## 参考

- [仓库 README](../../skills/codex-claude-academic-skills/README.md)
- [知乎文章一](https://www.zhihu.com/question/2058648818057531853/answer/2065455008460624405)
- [知乎文章二](https://www.zhihu.com/question/2058648818057531853/answer/2063668103637210722)
