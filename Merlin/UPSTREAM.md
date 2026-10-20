# Merlin 来源与版本

## 固定源码

| 项目 | 记录 |
| --- | --- |
| 原作者仓库 | https://github.com/StanfordMIMI/Merlin |
| 固定提交 | `e1552df771367aa2589ec7dcfd3151dc91f8ce74` |
| 提交日期 | 2026-09-24 |
| 本次收录日期 | 2026-10-20 |
| 包版本 | `merlin-vlm 0.0.7`，按本次 `pyproject.toml` |
| 源码许可证 | [MIT](upstream/LICENSE)，Copyright 2025 Stanford MIMI Lab |
| 文件数量 | 44 个受版本控制文件 |
| 字节总量 | 3,317,928 |

固定版本入口：[GitHub 提交](https://github.com/StanfordMIMI/Merlin/commit/e1552df771367aa2589ec7dcfd3151dc91f8ce74)。

`upstream/` 来自该提交的 `git archive`，逐文件保留原始内容，包括 README、LICENSE、锁文件、示例和原目录结构。没有嵌套 `.git`，没有导入上游完整提交历史。上游历史可通过原仓库和固定提交追溯；GitHub `project` 仓库记录本次收录和后续中文材料修改。

文件的 SHA-256 与大小保存在 [UPSTREAM_MANIFEST.json](UPSTREAM_MANIFEST.json)。这个清单适用于未修改的快照；今后若改源码，应明确记录补丁或更新到另一个固定提交，不应继续声称与此版本相同。

## 中文材料

根 README 与 `docs/01` 至 `docs/04` 是根据当前源码整理的介绍、阅读指南和研究思考，不属于原作者官方说明。`docs/05-paper-notes.md` 来自已有的中文论文精读，已将 Obsidian 链接改成可在 GitHub 阅读的引用，并去掉依赖本地附件的图片引用。

论文精读针对 **arXiv:2406.06512v2**；官方仓库当前引用 Nature 论文，DOI 为 [10.1038/s41586-026-10181-8](https://doi.org/10.1038/s41586-026-10181-8)。笔记中的页码、表号和数据差异仍按 v2 记录，不代表已逐项对照正式发表版。

## 运行资源

模型权重和示例 CT 由上游代码按需下载，本目录不包含它们。报告生成还依赖 RadLLaMA-7B。数据与模型资源的使用条件应分别以其来源为准，不把源码 MIT 许可证自动等同于数据或全部权重的许可。

[Merlin-nnUNet](https://github.com/ashwinkumargb/Merlin-nnUNet) 是独立仓库，本次只提供链接，没有把它作为已核验源码打包。

## 更新方式

1. 从官方仓库选定新提交，先读更新内容。
2. 在独立临时目录导出新快照，对比旧版本。
3. 更新 `upstream/`、这份版本记录和逐文件清单。
4. 重新检查源码笔记中的输出形状、依赖、模式开关与预处理。
5. 验证文档链接后，再提交到本仓库。

本次完成了快照一致性、Python 语法和中文文档相对链接核验；未安装依赖、下载模型或运行推理。
