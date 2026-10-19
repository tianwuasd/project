# 科研 Skill 补全与介绍文档 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 补下载两篇文章提到但本地缺失的科研 skill，并在 `docs/skill介绍/` 建立覆盖 12 个 skill 的中文介绍文档与总索引。

**Architecture:** 保留每个上游仓库在 `skills/` 下的独立目录；文档与仓库解耦，使用一个总览 `README.md` 和 11 个单 skill Markdown 文件。介绍内容以仓库公开 README/skill 文件为事实来源，并单独标注知乎文章中的主观评价与未独立验证的结论。

**Tech Stack:** Git、Markdown、PowerShell/Bash 文件操作。

---

### Task 1: 补下载缺失仓库

**Files:**
- Create: `skills/academic-research-skills-codex/`
- Create: `skills/codex-claude-academic-skills/`
- Create: `skills/claude-scholar/`
- Create: `skills/auto-empirical/`
- Create: `skills/nature-skills/`

- [x] **Step 1: 使用用户提供或文章中的仓库地址克隆缺失项目**

```bash
git clone <verified-repository-url> skills/<directory-name>
```

只克隆尚未存在的项目；不修改已有仓库。

- [x] **Step 2: 验证每个新目录是 Git 仓库**

```bash
git -C skills/<directory-name> rev-parse --is-inside-work-tree
```

Expected: 每个目录输出 `true`。

### Task 2: 建立文档目录和总索引

**Files:**
- Create: `docs/skill介绍/README.md`

- [x] **Step 1: 写入总索引**

总索引列出 11 个 skill 的中文名称、主要方向、文章评价（如有）、本地目录和对应介绍页链接，并明确说明评价来自用户提供的知乎文章，不等同于独立测评。

- [x] **Step 2: 写入选择速查表**

按“文献管理/引用验证、论文写作、科学计算、AI/深度学习、社科实证、无人值守改稿、论文视觉化”分类，给出适用场景和组合建议。

### Task 3: 编写单 skill 介绍页

**Files:**
- Create: `docs/skill介绍/academic-research-skills.md`
- Create: `docs/skill介绍/scientific-agent-skills.md`
- Create: `docs/skill介绍/auto-claude-code-research-in-sleep.md`
- Create: `docs/skill介绍/ai-research-skills.md`
- Create: `docs/skill介绍/research-paper-writing-skills.md`
- Create: `docs/skill介绍/wenyu-chiou-ai-research-skills.md`
- Create: `docs/skill介绍/paper-spine.md`
- Create: `docs/skill介绍/paper-craft-skills.md`
- Create: `docs/skill介绍/codex-claude-academic-skills.md`
- Create: `docs/skill介绍/claude-scholar.md`
- Create: `docs/skill介绍/auto-empirical.md`
- Create: `docs/skill介绍/nature-skills.md`

- [x] **Step 1: 为每个 skill 按统一模板写中文介绍**

每页包含：来源与本地目录、项目定位、核心功能、适用场景、优势、局限与风险、知乎文章评价、与其他 skill 的搭配、使用前核查事项。事实描述引用本地仓库公开文件；找不到的信息明确写“仓库资料未说明”，不补造功能。

- [x] **Step 2: 统一评价口径**

将“夯/顶级/人上人/NPC”等文章评级放在“文章评价”小节，并注明这是文章作者的体验性判断；对引用验证、联网检索、自动运行等涉及可靠性的能力，提醒用户自行复核输出。

### Task 4: 文档自检

**Files:**
- Verify: `docs/skill介绍/README.md`
- Verify: `docs/skill介绍/*.md`

- [x] **Step 1: 核对 11 个 skill 均有独立介绍页和索引链接**

```bash
ls docs/skill介绍
```

Expected: `README.md` 加 12 个 skill 介绍页。

- [x] **Step 2: 核对缺失仓库均已补全**

```bash
for dir in skills/*; do test -d "$dir/.git" && echo "$dir: ok"; done
```

Expected: 新补项目和原有项目均有对应 Git 元数据。
