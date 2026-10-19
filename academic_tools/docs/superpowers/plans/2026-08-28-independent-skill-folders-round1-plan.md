# 科研 Skill 独立目录第 1 轮 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Academic Research Skills Claude 版、ARS-Codex、Claude Scholar 建立独立详细文档目录，并提供只使用当前本地源码的操作指南。

**Architecture:** 保留 `docs/skill文档/*.md` 旧文档不动；为每个目标 skill 新建同名目录，目录内固定包含导航、功能、使用、源码构建、本地使用、验证风险六个 Markdown 文件。源码分析直接链接 `skills/<repo>` 下的真实路径，构建说明区分 Markdown 加载、插件打包、脚本和可选依赖，不虚构统一编译流程。

**Tech Stack:** Markdown、Claude Code/Codex skill 格式、Git 仓库源码、PowerShell/Bash。

---

### Task 1: 建立三个独立目录

**Files:**
- Create: `docs/skill文档/academic-research-skills-claude/`
- Create: `docs/skill文档/ars-codex/`
- Create: `docs/skill文档/claude-scholar/`

- [ ] **Step 1: 创建目录并确认旧文件保留**

```bash
mkdir -p docs/skill文档/academic-research-skills-claude docs/skill文档/ars-codex docs/skill文档/claude-scholar
```

Expected: 三个新目录存在，原有 `docs/skill文档/academic-research-skills-claude.md`、`ars-codex.md`、`claude-scholar.md` 仍存在。

### Task 2: 编写每个 skill 的功能和使用文档

**Files:**
- Create: 每个目标目录的 `README.md`
- Create: 每个目标目录的 `01-功能说明.md`
- Create: 每个目标目录的 `02-详细使用指南.md`

- [ ] **Step 1: 写入目录导航**

每个 README 链接 5 个专题文件，并注明对应真实源码目录和本轮文档范围。

- [ ] **Step 2: 写入功能说明**

按仓库实际 workflow、入口、子 skill、输入、输出、适用场景和不适用场景说明，不把知乎评价写成源码事实。

- [ ] **Step 3: 写入详细使用指南**

包含前置检查、逐阶段提示词、输入文件结构、预期产物、人工确认点、失败处理和与其他 skill 的衔接。

### Task 3: 编写源码结构、构建和本地使用指南

**Files:**
- Create: 每个目标目录的 `03-源码结构与构建.md`
- Create: 每个目标目录的 `04-本地使用指南.md`

- [ ] **Step 1: 记录真实源码结构**

列出 `skills/<repo>` 下的入口、`SKILL.md`、agents、references、scripts、插件元数据和配置文件，并说明它们的关系。

- [ ] **Step 2: 记录构建/加载方式**

说明哪些内容是 Markdown 直接加载、哪些需要插件/CLI、哪些脚本或可选依赖会被调用；如果没有传统编译步骤，明确写出“无统一编译步骤”。

- [ ] **Step 3: 写入本地使用指南**

只引用当前项目的 `skills/<repo>` 路径，给出 Claude Code/Codex 读取真实 `SKILL.md` 的最小方式、输入准备和输出位置；禁止完整安装、全局复制、插件注册或自动安装依赖。

### Task 4: 编写验证与风险说明

**Files:**
- Create: 每个目标目录的 `05-验证与风险.md`

- [ ] **Step 1: 写入验证清单**

覆盖引用、数据、实验、文件写入、MCP/API、权限、输出一致性和人工审查。

- [ ] **Step 2: 写入失败路径**

为每个 skill 列出触发失败、依赖缺失、目录冲突、来源不可验证和跨模型/外部调用的处理方式。

### Task 5: 更新导航并验收第 1 轮

**Files:**
- Modify: `docs/skill文档/README.md`
- Modify: `docs/skill介绍/README.md`

- [ ] **Step 1: 记录第 1 轮和下一轮目标**

在详细文档导航中保留第 1–4 轮历史，新增独立目录第 1 轮记录；下一轮只列 2–3 个 skill。

- [ ] **Step 2: 核对目录文件数量**

```bash
python -c "from pathlib import Path; root=Path('docs/skill文档'); names=('academic-research-skills-claude','ars-codex','claude-scholar'); assert all(len(list((root/name).glob('*.md')))==6 for name in names); print('folder file count passed')"
```

Expected: 三个目录各有 6 个 Markdown 文件；旧扁平文件仍存在。

- [ ] **Step 3: 核对本地路径和占位符**

确认三个 `04-本地使用指南.md` 均链接到对应 `skills/` 目录，所有 18 个新文件不含 `TBD`、`TODO` 或“稍后补充”。
