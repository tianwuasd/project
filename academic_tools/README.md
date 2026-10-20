# 学术科研 Skill 合集

本目录汇集 13 个上游科研技能仓库和配套中文使用说明，覆盖文献管理、选题、实验、科研计算、论文写作与投稿。源码已于 **2026 年 10 月 20 日（北京时间）**逐仓库核对上游默认分支：**2 个仓库更新，11 个原本已是最新**。

## 从这里开始

- [科研 Skill 介绍与选型](docs/skill介绍/README.md)
- [中文详细使用文档](docs/skill文档/README.md)
- [本次更新记录与版本变化](UPDATE-2026-10-20.md)
- [首次完整同步记录](UPDATE-2026-10-19.md)
- [来源、完整提交号与文件统计](SOURCES.json)

## 当前源码快照

`SKILL.md` 数量按提交树中的实际文件统计，包含分发副本，不表示去重后的独立功能数量。

| 仓库与源码入口 | 本次结果 | 固定提交 | SKILL.md 文件数 |
| --- | --- | --- | ---: |
| [academic-research-skills](skills/academic-research-skills/README.md) | 已是最新 | [cfd081d8](https://github.com/Imbad0202/academic-research-skills/commit/cfd081d8f309ec65254d9f9f04c845ee78b1ec1d) | 5 |
| [academic-research-skills-codex](skills/academic-research-skills-codex/README.md) | 已是最新 | [71528d93](https://github.com/Imbad0202/academic-research-skills-codex/commit/71528d9349bcdacfbe94111f06d9e2d82215a44d) | 2 |
| [AI-Research-SKILLs](skills/AI-Research-SKILLs/README.md) | 已是最新 | [773a5294](https://github.com/Orchestra-Research/AI-Research-SKILLs/commit/773a52944ba4747a18bd4ae9ade53fff041adcbc) | 98 |
| [Auto-claude-code-research-in-sleep](skills/Auto-claude-code-research-in-sleep/README.md) | 已是最新 | [43a5daf6](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/commit/43a5daf6f8a2db7b87b48cb51d45c6ddb3c9a6d3) | 193 |
| [auto-empirical](skills/auto-empirical/README.md) | 已是最新 | [09097ed9](https://github.com/brycewang-stanford/Auto-Empirical-Research-Skills/commit/09097ed9533ff9155cc1b64dd16b779d5e323074) | 1172 |
| [claude-scholar](skills/claude-scholar/README.md) | 已是最新 | [d394fb59](https://github.com/Galaxy-Dawn/claude-scholar/commit/d394fb599ed9e6ea2212a9e6daa864345fe97a26) | 45 |
| [codex-claude-academic-skills](skills/codex-claude-academic-skills/README.md) | 已是最新 | [7ed6377f](https://github.com/zLanqing/codex-claude-academic-skills/commit/7ed6377f0efb6a38951b48ef03b19d996e454b1f) | 24 |
| [nature-skills](skills/nature-skills/README.md) | 更新 26 个提交 | [b0de5a9e](https://github.com/Yuan1z0825/nature-skills/commit/b0de5a9ec3f8439433c742782f4fbbe582a253d7) | 23 |
| [paper-craft-skills](skills/paper-craft-skills/README.md) | 已是最新 | [3be47a2a](https://github.com/zsyggg/paper-craft-skills/commit/3be47a2a53cc35a411c587bca5231a08de57287a) | 3 |
| [PaperSpine](skills/PaperSpine/README.md) | 已是最新 | [42312b8d](https://github.com/WUBING2023/PaperSpine/commit/42312b8d9c76ab25990cddfd76e3622276854118) | 7 |
| [Research-Paper-Writing-Skills](skills/Research-Paper-Writing-Skills/README.md) | 已是最新 | [77e7c2c1](https://github.com/Master-cai/Research-Paper-Writing-Skills/commit/77e7c2c1ba06f7d71844873147665437a03aac1b) | 1 |
| [scientific-agent-skills](skills/scientific-agent-skills/README.md) | 已是最新 | [593377bb](https://github.com/K-Dense-AI/scientific-agent-skills/commit/593377bb048752abb0152deab9fbb1083a6087df) | 177 |
| [WenyuChiou-ai-research-skills](skills/WenyuChiou-ai-research-skills/README.md) | 更新 3 个提交 | [22ac7e0c](https://github.com/WenyuChiou/ai-research-skills/commit/22ac7e0c91fcc8208bb86b1167a7fe009d397642) | 0 |

合计保留 **20,344 个上游文件**、5 个符号链接（包含在文件总数内）和 5 个固定版本的子模块引用，源码内容约 **710.7 MB**。每个源码子目录的 Git 树与对应上游提交一致；原许可证和来源说明随源码保留。

## 使用说明与版本边界

中文说明保留此前整理的工作流和示例，本次更新了版本标记、实际源码统计及已失效的本地链接；没有把全部旧版指南重新做功能测试。安装入口、命令和依赖以表中固定版本的上游 README 为准。

WenyuChiou 的 v1.7.7 目录声明 17 个可移植 skill、5 个插件，但该仓库提交树实际包含 **0 个 SKILL.md 文件**。具体 skill 源码由其 [skill-directory.md](skills/WenyuChiou-ai-research-skills/docs/skill-directory.md) 指向其他仓库，不能把目录声明当成本地已展开的源码。

上游已有的 5 个嵌套 Git 引用保留原提交号。项目根目录的 `.gitmodules` 提供对应路径和来源：Auto-Empirical 的 1 项来自其 `.gitmodules`，Claude Scholar 的 4 项来自其 `plugins/known_marketplaces.json`；这些外部依赖没有在本次快照中展开或运行。

本次完成源码同步、内容完整性与文档链接核验；未执行上游安装脚本、模型/API 调用或研究实验，也未更改本机已安装的 Codex 技能。
