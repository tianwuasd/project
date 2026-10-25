# 继续研究提示

最新进展覆盖下方旧研究阶段：用户已要求并实施 nnU-Net B0 主体代码、server-engineering 中文任务面板与 GitHub 交付。先读 docs/NNUNET_BASELINE.md、docs/SERVER_PANEL.md 和 docs/BASELINE_VALIDATION.md，不要重写已经存在的基线。默认8 fit+4 cal监督、selector4/U22预留；尚未开始真实病例GPU训练。VISTA/专家/点击仍属后续机制，不应混称已实现。仓库当前 private，更新不用公共镜像。

请接手 `D:/code_project/vista3d_local_expert_refinement`。先读根目录 AGENTS.md、.codex/HANDOFF.md、.codex/STATE.json 和 .codex/DECISIONS.md，再读 docs/research_framework.md 和 docs/assessment-2026-10-25.md。

研究核心是：主体分割模型和局部专家在相同来源中心材料上学习；测试时局部专家保留并提出局部分割意见；主体分割模型选择采纳，通过 VISTA3D 正负点击修正自己的分割，最终输出主体分割模型掩膜。来源有少量标注和额外未标注 CT，目标中心 CT 目前用于预测。两名局部专家、单轮起步及测试固定参数是首版建议，不要误认成用户永久限制。

本项目已经完成文献整理，42 项明确提及的论文、17 篇 PDF、18 份既有笔记，以及正确 Academic Agent 的完整研究快照均在项目内。不要从头重复检索或再次误用 ARS 代替 testing_tools/academic_agent。

当前没有模型实验结果。请先恢复已有数据说明，核对真实标签、病例划分、主体分割模型检查点与硬件条件，按 experiments/PLAN.md 优先推进局部专家互补性和实际点击收益验证。不能把框架建议或文献工具的复核统计写成分割性能结果。

本轮已将当前称呼改为主体分割模型、局部专家、修正采纳模块、交互修正器。单专家先行是新增建议，优先验证实际点击收益；历史资料和固定快照不改写。
