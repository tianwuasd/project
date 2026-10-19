# PaperSpine 文档目录

> 版本说明（2026-10-19）：源码已更新；正文保留此前整理的指南，功能与安装入口以当前上游 README 为准。见[本次更新记录](../../../UPDATE-2026-10-19.md)。

本目录说明 `skills/PaperSpine/` 的贡献优先论文生产流程、源码组织和本地受限使用方式。

## 文档导航

1. [功能说明](01-功能说明.md)
2. [详细使用指南](02-详细使用指南.md)
3. [源码结构与构建](03-源码结构与构建.md)
4. [本地使用指南](04-本地使用指南.md)
5. [验证与风险](05-验证与风险.md)

## 源码入口

- 主入口：`skills/PaperSpine/src/skill/SKILL.md`
- 源码目录：`skills/PaperSpine/src/`
- 旧版完整说明：[00-原始完整说明.md](00-原始完整说明.md)

PaperSpine 以一个主 skill 协调阶段性产物；具体阶段资源、agents、references 和 scripts 位于 `src/` 下。
