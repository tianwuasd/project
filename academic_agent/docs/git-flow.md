# Git Flow

应用作为 `testing_tools/academic_agent/` 整体目录维护；仓库级 CI 是目录之外唯一的应用基础设施。

1. `main` 承载已通过验收的正式版；`develop` 集成下一版。
2. 从 `develop` 建立 `feature/<功能名>`，在隔离工作区开发、审查与修复。
3. 功能测试与独立审查通过后，用非快进合并将 feature 集成至 develop。
4. 从 develop 建立 `release/academic-agent-<版本>`，冻结功能并执行完整测试、安装包使用自检与相应真实接口验证。
5. release 非快进合并 main 并回合 develop；在正式提交添加附注标签 `academic-agent-v<版本>`。
6. 推送并核对远端分支、标签、提交及目录；检查六环境 CI。未完成的阶段不能仅因已有代码就标记发布。

0.1.0 与 0.2.0 分别沿对应的 feature、release 和附注标签维护。0.2.0 的分支为 `feature/research-usability-v02` 和 `release/academic-agent-0.2.0`，标签为 `academic-agent-v0.2.0`；下一阶段使用 `feature/innovation-v03`，具体验收见 [持续路线](project-roadmap.md)。

用户已授权研发和上传。禁止强制推送及提交本地研究数据。后续功能继续 feature/*，正式版紧急修复用 hotfix/*，同时回合 main 和 develop。`git log --graph --oneline --all` 可检查实际流程；发行标签是可复现入口。
