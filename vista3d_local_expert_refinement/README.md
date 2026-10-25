# 全局老师与局部学生：VISTA3D 选择性点击修正

创建日期：2026-10-25。阶段：文献研究与框架设计完成；尚未实现或训练分割系统。

## 研究问题

来源中心有少量标注 CT 和额外未标注 CT。老师与学生学习相同的当前来源材料，老师关注整体，学生分别强化局部结构。在目标中心预测时，学生继续提出局部分割意见；老师系统判断是否采纳，将有益意见转成三维正负点击，通过 VISTA3D 交互分支更新老师掩膜。最终输出是修正后的老师分割。

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
docs/                    当前研究方案与对话整理
literature/              文献索引；17 篇 PDF；本地精读与批注
research/                Academic Agent 项目数据库、原文、证据及固定导出快照
experiments/              最小实验计划与待填写的数据条件表；没有虚构结果
archive/                 原讨论和原笔记的保真副本；包含已过时的历史解释
tools/                   本次归档脚本和只读完整性检查
AGENTS.md                后续助手的读取顺序与项目工作约定
```

## 已完成与未完成

- 实际使用 `D:/projects/testing_tools/academic_agent` 的 Academic Agent 0.4.0，已完成 12 篇全文阅读与复核。它与 `academic_tools` 中的 ARS 是不同项目。
- 本项目保存完整研究快照：61 条书目、12 篇阅读卡、195 条提取陈述、检索日志和证据。193 条支持、2 条部分支持是模型复核状态，不是人工准确率。
- 已将原有研究文件复制归档；原 Obsidian 文献库及原 Agent 项目仍保留。
- 尚未核定标签结构、病例数量、患者划分、硬件、目标中心评价集，也未运行 CT 推理或训练。
- 测试时固定参数是推荐的第一版协议；用户已确认的是目标中心 CT 用于预测。不要把建议提升成用户永久禁止其他协议。

## 后续入口

在这个目录继续工作时，先读 `AGENTS.md` 和 `.codex/HANDOFF.md`。优先核对已有数据说明与标签清单，再填写 [数据条件表](experiments/data_requirements.json)，随后执行互补性与点击收益实验。

GitHub 同步目标是既有仓库 [tianwuasd/project](https://github.com/tianwuasd/project) 的 [vista3d_local_expert_refinement](https://github.com/tianwuasd/project/tree/main/vista3d_local_expert_refinement) 子目录，包含本项目完整研究归档。同步范围与授权见 [.codex/GITHUB_SYNC.md](.codex/GITHUB_SYNC.md)。本地原项目仍保留，尚未注册 Codex 侧边栏项目。学术 Agent 软件与运行环境的现有位置、研究快照恢复方式见 [.codex/ENVIRONMENT.md](.codex/ENVIRONMENT.md)。
