# 局部专家协同的 VISTA3D 选择性分割修正

创建日期：2026-10-25。当前已实现 **wholeheart CT 的 nnU-Net 监督基线 B0 与服务器任务面板**。已完成合成数据 CPU 集成验证；尚未进行真实病例 GPU 训练或取得性能指标。

用户确认先以 nnU-Net 建立 baseline。默认 A+B→G：8 fit + 4 cal 实际参与监督训练；4 selector、22 U 为后续研究预留。VISTA3D、局部专家与点击修正仍是后续阶段。

- [工程与方法说明](docs/NNUNET_BASELINE.md)：预处理、训练、测试、重要公式、模块和数据边界。
- [服务器面板与命令](docs/SERVER_PANEL.md)：运行 `bash start_server.sh`，命名任务、短测、训练、续跑、查询和更新。
- [面板 v0.2.0 验证](docs/PANEL_TEMPLATE_VALIDATION.md)：CT 母版适配、环境安装子阶段、持续进度、选择日志和任务进程树。
- [验证记录](docs/BASELINE_VALIDATION.md)：已测范围与尚需服务器验证的部分。
- [wholeheart 初步研究方案](docs/wholeheart_initial_plan_2026-10-25.md)：后续专家／采纳／交互目标的来源；其中“只出计划”是历史阶段。

## 研究问题

来源中心有少量标注 CT 和额外未标注 CT。主体分割模型与局部专家使用相同的当前来源病例池，分别学习整体分割与局部专长。预测时，局部专家持续提出建议，修正采纳模块选择预计有益的三维正负点击，交互修正器执行并更新主体掩膜。最终输出为修正后的主体分割。

本机工作目录：`D:/projects/vista3d_local_expert_refinement`。角色命名与评估见 [项目评估](docs/assessment-2026-10-25.md)。历史对话与固定研究快照保留原称呼。

## 从这里阅读

1. [交接说明](.codex/HANDOFF.md)：当前状态、不可丢失的用户意图、下一步。
2. [对话整理](docs/conversation_context.md)：需求演变与关键原话，非完整逐字聊天导出。
3. [最新研究方案](docs/research_framework.md)：相似工作、训练方法、架构取舍与创新边界。
4. [最小实验计划](experiments/PLAN.md)：互补性 → 点击可达收益 → 采纳能力 → 跨中心评价。
5. [文献总索引](literature/README.md)：42 项明确提及的论文，17 篇 PDF、18 份已有笔记。
6. [Academic Agent 阅读卡](research/academic_agent_snapshot/exports/v-2026-10-25T054638-996790-0000-0128989e/reading-cards.md)及[比较表](research/academic_agent_snapshot/exports/v-2026-10-25T054638-996790-0000-0128989e/comparison.md)。

## 项目结构

```text
.codex/                  交接、决策、环境、下一步、状态和文件校验清单
05_src/                  nnU-Net基线计算引擎与标准库任务面板
06_configs/              固定依赖、协议说明和GPU配置
tests/                   合成回归与显式CPU集成验证
docs/                    工程方法、服务器命令、研究方案与对话整理
literature/              文献索引；17 篇 PDF；本地精读与批注
research/                Academic Agent 项目数据库、原文、证据及固定导出快照
experiments/              最小实验计划与待填写的数据条件表；没有虚构结果
archive/                 原讨论和原笔记的保真副本；包含已过时的历史解释
tools/                   本次归档脚本和只读完整性检查
AGENTS.md                后续助手的读取顺序与项目工作约定
```

## 已完成与未完成

- 原研究实际使用来源机器上的 `D:/projects/testing_tools/academic_agent` 的 Academic Agent 0.4.0，已完成 12 篇全文阅读与复核。它与 `academic_tools` 中的 ARS 是不同项目。
- 本项目保存完整研究快照：61 条书目、12 篇阅读卡、195 条提取陈述、检索日志和证据。193 条支持、2 条部分支持是模型复核状态，不是人工准确率。
- 已将原有研究文件复制归档；原 Obsidian 文献库及原 Agent 项目仍保留。
- 已核对 wholeheart 七类标签与清单：106 组记录均有文件，沿用隔离后为 CT 58/MRI 43。少标注子划分已在 cohort.py 固定；独立患者身份、实际服务器 GPU 条件和七类交互接口仍待核定。没有真实病例模型结果。
- 测试时固定参数是推荐的第一版协议；用户已确认的是目标中心 CT 用于预测。不要把建议提升成用户永久禁止其他协议。

## 后续入口

在这个目录继续工作时，先读 `AGENTS.md` 和 `.codex/HANDOFF.md`。按 [服务器面板](docs/SERVER_PANEL.md) 先完成 B0 真实数据短测和训练；保留数据边界，之后推进互补性与点击收益实验。

面板v0.3的GPU/路径、c/u/w/v及子任务内部执行逐项对应见 [CT完整母版严格验收](docs/CT_STRICT_AUDIT.md)。

GitHub 同步目标是既有仓库 [tianwuasd/project](https://github.com/tianwuasd/project) 的 [vista3d_local_expert_refinement](https://github.com/tianwuasd/project/tree/main/vista3d_local_expert_refinement) 子目录，包含本项目完整研究归档。同步范围与授权见 [.codex/GITHUB_SYNC.md](.codex/GITHUB_SYNC.md)。本地原项目仍保留，尚未注册 Codex 侧边栏项目。学术 Agent 软件与运行环境的现有位置、研究快照恢复方式见 [.codex/ENVIRONMENT.md](.codex/ENVIRONMENT.md)。
