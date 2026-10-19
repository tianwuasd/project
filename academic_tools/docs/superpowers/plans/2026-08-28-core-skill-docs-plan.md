# 第一轮核心科研 Skill 使用文档 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 建立 `docs/skill文档/` 详细使用文档层，并完成 Academic Research Skills Claude 版、ARS-Codex、Claude Scholar 三个核心 skill 的可执行说明。

**Architecture:** `docs/skill介绍/` 继续保存短篇定位和选型信息；`docs/skill文档/` 保存按统一模板编写的深度使用指南。每篇指南只描述仓库 README、`SKILL.md` 和已有官方文档确认的功能，并把 Claude Code/Codex 的安装方式、输入准备、典型工作流、输出核查和风险分开说明。

**Tech Stack:** Markdown、Git 仓库本地源码、Claude Code/Codex 使用说明。

---

### Task 1: 建立详细文档目录规范

**Files:**
- Create: `docs/skill文档/README.md`

- [x] **Step 1: 写入文档导航和统一模板**

README 必须说明本轮覆盖的三个 skill、后续增量轮次的命名规则，并固定每篇指南包含：定位、前置条件、安装、输入准备、操作步骤、典型提示词、预期产物、结果核查、失败/风险、与其他 skill 的衔接和来源。

- [x] **Step 2: 写入增量推进记录**

记录本轮完成项、下一轮候选项和“未验证功能不作保证”的规则；不要把文档当作自动安装脚本。

### Task 2: 完成 Academic Research Skills Claude 版指南

**Files:**
- Create: `docs/skill文档/academic-research-skills-claude.md`

- [x] **Step 1: 根据本地仓库确认安装入口和工作流**

覆盖 Claude Code 插件/skill 入口、五类工作流、引用与证据门禁、需要用户提供的材料，以及最小可执行提示词。

- [x] **Step 2: 写入逐步使用示例**

至少包含研究问题收敛、引用核验、论文审稿三个例子；每个例子列出输入、期望产物和人工检查点。

- [x] **Step 3: 写入风险与边界**

明确不能验证的引用/数据必须保持未验证，外部网络和脚本调用需要用户确认，Claude 版与 ARS-Codex 不可混装。

### Task 3: 完成 ARS-Codex 指南

**Files:**
- Create: `docs/skill文档/ars-codex.md`

- [x] **Step 1: 根据 README 记录 Codex 安装和路由**

覆盖插件安装、直接 skill 安装、`academic-research-suite` 路由、五类工作流和 `ars-*` 别名。

- [x] **Step 2: 写入分阶段操作示例**

至少包含 Socratic 研究问题收敛、论文大纲/引用检查和完整 pipeline 分阶段执行；说明每步应暂停并核查什么。

- [x] **Step 3: 写入 Codex 特有边界**

说明 vendored Claude 文件、Codex router、hooks、cross-model verification、缓存/脚本引用检查的关系，避免把“自动”理解成无条件外部调用。

### Task 4: 完成 Claude Scholar 指南

**Files:**
- Create: `docs/skill文档/claude-scholar.md`

- [x] **Step 1: 根据 README 记录安装与分支选择**

覆盖 Claude Code 主分支、Codex/Kimi/OpenCode 分支、全量/最小/选择性安装，以及 Git、Zotero、Obsidian 等可选依赖。

- [x] **Step 2: 写入文献到论文的操作流程**

覆盖选题、Zotero 导入与整理、结构化文献笔记、Obsidian 知识库、实验结果分析、报告和论文/审稿回复，并给出可复制提示词。

- [x] **Step 3: 写入集成核查和风险**

明确 Zotero/Obsidian/MCP 权限、分支不混用、本地 Markdown 合并、原文核验和人类决策边界。

### Task 5: 第一轮文档核验

**Files:**
- Verify: `docs/skill文档/*.md`

- [x] **Step 1: 核对文件和必需章节**

```bash
ls docs/skill文档
```

Expected: `README.md`、`academic-research-skills-claude.md`、`ars-codex.md`、`claude-scholar.md`；三篇指南均包含“安装”“典型提示词”“结果核查”“风险”。

- [x] **Step 2: 核对本地仓库链接**

确认三个指南分别链接到 `skills/academic-research-skills/`、`skills/academic-research-skills-codex/` 和 `skills/claude-scholar/`，并且目标目录存在。

- [x] **Step 3: 核对占位符和增量记录**

```bash
python -c "from pathlib import Path; text='\\n'.join(p.read_text(encoding='utf-8') for p in Path('docs/skill文档').glob('*.md')); assert not any(x in text for x in ('TBD', 'TODO', '稍后补充')); print('placeholder check passed')"
```

用文本搜索确认新增文档不含 `TBD`、`TODO` 或空泛的“稍后补充”，并在 `docs/skill文档/README.md` 记录本轮完成状态。
