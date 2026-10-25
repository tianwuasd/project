# 项目工作约定

1. 开始时依次阅读 `.codex/HANDOFF.md`、`.codex/STATE.json`、`.codex/DECISIONS.md`；做研究前再读 `docs/research_framework.md` 与 `literature/README.md`。
2. 保留用户真实目标：同源训练、局部学生在测试时继续提出建议、老师选择采纳、点击修正老师、最终输出老师掩膜。不能自行退回“仅训练时伪标签蒸馏、最终只部署学生”的设定。
3. 区分用户确认、助手建议和待实验验证的假设。两名学生、固定参数、点击前收益预测是首版建议，不是已实施结果或不能修改的限制。
4. 当前没有模型实验结果。所有数值必须附真实实验记录；不能将学术 Agent 阅读复核计数当作模型性能或人工文献准确率。
5. 最新设计以 `docs/research_framework.md` 为准。`archive/` 为保真历史资料，其中一些角色解释、工具识别和访问诊断已过时；按 `.codex/DECISIONS.md` 处理冲突。
6. 使用学术 Agent 时定位 `D:/projects/testing_tools/academic_agent`，不要与 `D:/projects/project/academic_tools` 的 ARS 混淆。现有研究快照位于 `research/academic_agent_snapshot`；续做应复制到工作目录并保留固定快照。
7. 先恢复已有上下文，再询问缺失信息。常规本地整理和检查直接继续；没有必要重新请求用户已经给予的授权。用户于 2026-10-25 明确要求将本研究项目上传到 GitHub 的 `tianwuasd/project`，本次完整归档同步在此授权内；这不扩展到其他项目资料或长时间 GPU 训练。
8. 分割数据、训练/验证划分与患者信息另行管理。本归档没有 CT 数据；后续不得从文献中的示例臆测本项目类别和数量。本次授权覆盖已整理的研究讨论、文献、数据库与交接记录；以后新增的患者信息、凭据或其他私人材料不能据此自动公开。
9. 文献结论必须与阅读范围一致；保留摘要/片段/全文区别。早期资料不是当前自动核验结果。不要为凑新颖性而忽略 CGS、DnR、SegAgent、DeferredSeg、NeuroRefiner 等近邻。
10. 后续实质进展同步更新 `.codex/STATE.json`、`.codex/HANDOFF.md` 与 `.codex/DECISIONS.md`，注明哪些是用户确认、哪些仍是建议。保持 UTF-8，尽量使用项目内相对链接。
