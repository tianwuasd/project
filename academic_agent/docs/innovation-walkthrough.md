# 创新候选使用指南

创新工作流从当前已纳入、已阅读且已复核的证据生成可证伪的系统假设。候选、反例检索和评估都保存版本与输入指纹；研究材料变化后，`innovation-status` 会区分仍可追溯的历史候选、当前前提和已过期评估。

## 原创离线演练

以下演练只使用包内原创 A/B 合成材料和固定 Demo 协议，不联系论文来源或真实模型。输出目录与项目目录都必须是新的：

```powershell
uv run academic-agent --project ".runs/创新 演练" demo
uv run academic-agent --project ".runs/创新 演练" status
uv run academic-agent --project ".runs/创新 演练" innovate --work-id work_A --work-id work_B
uv run academic-agent --project ".runs/创新 演练" assess-innovation innovation_替换为实际ID
uv run academic-agent --project ".runs/创新 演练" innovation-status
```

请用 `status` 返回的实际 `work_id` 和 `innovate` 返回的实际候选 ID 替换占位符。未知 ID 会被拒绝并返回非零退出码。Demo 只接受固定原创材料；混入其他内容会明确拒绝，不应把这种拒绝解释为真实材料的学术判断。

也可以一次运行安装后自检：

```powershell
uv run academic-agent self-test --output .runs/usage-v03
```

自检会实际启动 CLI 子进程，检查一个候选、生成和评估文件、合成标签、独立复核状态、预算暂停后的缓存恢复，以及研究输入变化后的过期状态。报告明确记录未联系论文来源或真实模型，也不把固定 Demo 结果称为全球新颖性验证。

## 真实研究顺序

先完成普通研究流程，确保选择范围内的材料已有当前阅读卡和复核结果。然后按下面的顺序执行：

```powershell
uv run academic-agent --project projects/study guide
uv run academic-agent --project projects/study innovate --work-id work_A --work-id work_B
uv run academic-agent --project projects/study innovation-status
uv run academic-agent --project projects/study challenge innovation_实际ID --round 1
```

每个候选最多两轮反例检索，每轮最多两条定向查询，每个来源对每条查询最多保留十条结果。来源限流或部分失败时，命令保留已完成收据并返回非零状态；修复来源后重复同一候选和轮次即可恢复，不会暗中开始新一轮。

检索得到书目不等于已获得反例证据。逐篇查看候选记录，保留真实筛选理由，并为要使用的材料提供可读内容：

```powershell
uv run academic-agent --project projects/study screen work_反例ID --decision include --reason "与候选的关键前提直接相关"
uv run academic-agent --project projects/study attach work_反例ID "D:/papers/counterexample.pdf"
uv run academic-agent --project projects/study read --work-id work_反例ID
uv run academic-agent --project projects/study review --work-id work_反例ID
uv run academic-agent --project projects/study challenge innovation_实际ID --round 1
uv run academic-agent --project projects/study assess-innovation innovation_实际ID
uv run academic-agent --project projects/study innovation-status
```

PDF 必须有可提取文字层；扫描件显示 `needs_ocr` 时，先在外部完成 OCR，再关联得到的可读文件。Exchange 后端会给出待填写的请求文件；响应必须保持相同 fingerprint 并通过结构验证，再重复原命令。

评估分别保存新颖性线索、研究价值、可行性和证据强度，并对每个生成文本单元独立复核。命令可以在检索尚未开始、来源不完整或证据不足时完成结构化评估；这些缺口会保存在 `coverage_status`、`gaps` 和报告中。研究者仍需核对真实来源覆盖、学术意义和实验可行性，不能从候选或评估推断全球首次。

预算用尽时，提高 `project.json` 中的 `max_model_calls` 后重复返回的 `resume_argv`。相同输入的已完成模型尝试会复用；工具不会自动提高预算。研究材料、来源文件或候选上下文变化后，先运行 `innovation-status` 和 `guide`，再按给出的完整 argv 重新生成或评估。
