# Assistant2 Branch Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 助手2代码更新固定到独立发布分支及独立目录，保护助手1和冻结版本。

**Architecture:** 复用现有update.py的Git内容核验、锁和子目录同步，引入固定BRANCH并用于所有本地/远端ref。真实本地Git来源验证固定分支、普通更新、远端变化拒绝、活动锁保护及另一个代码目录保持不变。

**Tech Stack:** Python标准库、Git、现有代码读取锁。

**Spec:** ../specs/2026-11-15-assistant2-offline-design.md

## Global Constraints

- 固定分支codex/chd-assistant2-offline-preprocess-20261115；不切换助手1目录或修改main。
- 只通过文件编辑工具修改源码；测试使用真实Git，不mock网络或返回结果。
- 临时文件在已批准t17工作根内，同期200文件及8GiB；报告总量10MiB。
- 最终仅上传助手2分支；公开子树chd-segmentation和其他项目分别核对。
- 本计划只处理独立更新；离线公共数据适配在助手1最终合同确定后使用另一份对应实施计划。

## Review Focus

- 任何残留main字面ref不得成为助手2下载、校验或激活目标。
- --force也不能绕过本地分支检查，不能改变助手1目录。
- 更新后重新查询同一官方分支，分支推进时不得报告目标已同步。
- 代码运行入口仍读取正确Git目录并持有原更新锁。
- 代码包内容和发布回执准确对应助手2分支，私有文件不进入树。

### Task 1: 固定分支及实际更新验收

**Files:** 修改update.py、AGENTS.md、docs/ASSISTANT2_PUBLICATION.md；新增tests/verify_assistant2_branch_update.py。

**Interfaces:** update.BRANCH为固定分支字符串；probe_source、GitRepository.remote_head/validate/update_project、synchronize和main统一使用该常量。源码目录仍为chd-segmentation；默认repo=/data4/zhangruichen/code/chd-segmentation-assistant2-repo，runtime=/data4/zhangruichen/course/chd-segmentation-assistant2；回执新增branch和origin_branch。

- [x] 编写真实Git短测：来源包含当前真实run.py、update.py、setup_env.py、pyproject.toml；创建独立同名分支和独立main代码目录。
- [x] 运行新增检查，观察缺少BRANCH或仍使用main时失败；保存失败日志并清理测试目录。
- [x] 将所有分支ref、默认目录和回执改为助手2常量，保留实际代码读锁和其他项目保护。
- [x] 同一真实Git测试核对更新、复用、活动锁和分支变化拒绝、错误本地分支拒绝及main目录文件/HEAD均未变化；实际run.py --code-info读取目标SHA。
- [x] 检查命令：python -B tests/verify_assistant2_branch_update.py --work-root WORK --report REPORT。Expected: passed，所有检查和清理回执完整。
- [x] 明确文件暂存并提交；最终离线适配的整体审阅同时核对此更新边界。
