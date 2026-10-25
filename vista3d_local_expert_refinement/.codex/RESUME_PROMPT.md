# 继续研究提示

请接手 `D:/projects/vista3d_local_expert_refinement`。先读根目录 AGENTS.md、.codex/HANDOFF.md、.codex/STATE.json 和 .codex/DECISIONS.md，再读 docs/research_framework.md。

研究核心是：老师和局部学生在相同来源中心材料上学习；测试时学生保留并提出局部分割意见；老师选择采纳，通过 VISTA3D 正负点击修正自己的分割，最终输出老师掩膜。来源有少量标注和额外未标注 CT，目标中心 CT 目前用于预测。两名学生、单轮起步及测试固定参数是首版建议，不要误认成用户永久限制。

本项目已经完成文献整理，42 项明确提及的论文、17 篇 PDF、18 份既有笔记，以及正确 Academic Agent 的完整研究快照均在项目内。不要从头重复检索或再次误用 ARS 代替 testing_tools/academic_agent。

当前没有模型实验结果。请先恢复已有数据说明，核对真实标签、病例划分、老师检查点与硬件条件，按 experiments/PLAN.md 优先推进学生互补性和实际点击收益验证。不能把框架建议或文献工具的复核统计写成分割性能结果。
