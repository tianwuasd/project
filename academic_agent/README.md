# Academic Agent · 学术工作 Agent V1

以 **Codex 驱动、证据可追溯、状态可恢复** 的本地学术工具。把关键词、文献记录和可读全文整理为阅读卡、比较表和带引用的综述工作稿。应用完整位于此目录，可独立安装。

本目录版本为 **0.4.0**，包含从文献发现、证据阅读、趋势与创新候选到跨文献写作、人工修订和领域扩展的本地工作流。离线安装验收与限量真实 Codex 观察已完成；真实整稿复核的 13 个支持、1 个证据不足决定均保留，不能作为学术准确率。发行文档在本地验收后冻结，正式分支、标签及 CI 状态以仓库记录为准，详见[0.4 验收记录](docs/validation-v0.4.md)。实验执行不在当前范围。

## 已实现能力

| 环节 | 功能 |
|---|---|
| 发现 | OpenAlex、Semantic Scholar、Crossref、arXiv；分页、限流/部分失败报告、原始响应与查询留档 |
| 管理 | RIS/BibTeX、DOI 规范化、保守去重、预印本版本、稳定引用键、具理由的纳入/排除 |
| 阅读 | 开放获取 PDF 下载；本地 PDF 文字层、Markdown、TXT；元数据/摘要/全文/需 OCR 区分 |
| 分析 | 问题、方法、数据、结果、局限、作者贡献提取；材料问答；页码/字符范围/哈希证据 |
| 复核 | 独立 Codex 会话检查完整可用上下文；支持/部分支持/矛盾/不足；具名人工复核；出版通知检查 |
| 写作准备 | 阅读卡、比较矩阵、证据缺口、提纲、带引用综述工作稿、BibTeX 与质量报告 |
| 可靠性 | SQLite、不可变原文快照、LangGraph 检查点、版本与租约校验、取消迟到结果、调用预算、缓存 |
| 指引与趋势 | 只读项目指引；冻结语料的年度与人工标签描述统计；离线哈希校验与完全重算 |
| 创新工作流 | 最多三个候选；两轮有界反例检索；新颖性线索、研究价值、可行性、证据强度四维评估与独立复核 |
| 扩展 | Source / Parser / Backend 协议；Codex、合成演示、手动 JSON 交换；可替换 API 后端 |

“作者贡献提取”不是自动证明创新性。综述工作稿按主题汇集已复核陈述，跨论文解释与论证仍需研究者综合。

## 安装与演示

需要 Python 3.11+，推荐 [uv](https://docs.astral.sh/uv/)。在 `academic_agent` 目录运行：

```powershell
uv sync --locked --extra dev
uv run academic-agent --project projects/demo demo
```

演示不需要模型或 API 密钥，使用两篇**原创合成研究样例**完成整个流程，产物明确标记合成数据。再次演示请使用新项目目录。

新用户可以从演示结果开始查看当前状态与下一步：

```powershell
uv run academic-agent --project projects/demo guide
uv run academic-agent --project projects/demo trends --date-basis record_year
uv run academic-agent verify-trends projects/demo/analyses/trends/替换为实际版本目录
```

`guide` 只读检查项目，不调用模型或论文网络，也不会自动把全部候选设为纳入；它返回中文说明、阻塞原因和保留空格与引号的 argv 数组。请逐篇检查其建议的筛选动作并填写真实理由。`trends` 只描述当前冻结语料，不推断领域总体增长；需要统一出版年份或方法标签时，使用 `--annotations annotations.json` 提供具名、带理由和来源的人工批注。趋势目录可由 `verify-trends` 在无项目数据库、网络或模型的情况下校验并重算。

也可标准安装：`python -m venv .venv`，然后 `.venv/Scripts/python.exe -m pip install -e ".[dev]"`。macOS/Linux 使用 `.venv/bin/python`。锁文件用于精确依赖重建；普通 pip 按兼容版本范围安装。

## 真实研究流程

先安装并登录 Codex，在自己的终端确认 `codex login status` 可用：

```powershell
uv run academic-agent doctor
uv run academic-agent --project projects/topic init --title "我的研究" --question "需要回答的研究问题" --query "graph neural network"
uv run academic-agent --project projects/topic run --stop-after discover
uv run academic-agent --project projects/topic status
```

检查候选，用返回的 `work_id` 完成筛选、关联全文后继续：

```powershell
uv run academic-agent --project projects/topic screen work_替换为真实ID --decision include --reason "研究问题与方法符合纳入范围"
uv run academic-agent --project projects/topic attach work_替换为真实ID "D:/papers/example.pdf"
uv run academic-agent --project projects/topic resume
```

已有开放获取链接可用 `fetch work_ID`。未附全文会使用摘要并始终显示 `abstract_only`；没有摘要或需要 OCR 时阅读明确失败。`screen --all` 批量应用同一决定，适用于你已检查整批候选的情况。

来源限流或暂时失败会保留已取得的文献，并暂停自动工作流；`resume` 可重试故障检索，不会无限自动重试。若某来源持续不可用，可在 `project.json` 暂时移除它，或用独立的 screen/read/review/export 命令处理已取得的部分语料。请保留查询日志中的来源缺口。

常用命令：

```powershell
uv run academic-agent --project projects/topic import bibliography.ris
uv run academic-agent --project projects/topic read --work-id work_替换为真实ID
uv run academic-agent --project projects/topic review
uv run academic-agent --project projects/topic ask "这些研究有哪些局限？"
uv run academic-agent --project projects/topic export
```

完成当前阅读与复核后，可生成候选、检索反例并评估。全局项目选项放在子命令前；`--work-id` 可重复，用于明确缩小完整输入范围：

```powershell
uv run academic-agent --project projects/topic innovate --work-id work_A --work-id work_B
uv run academic-agent --project projects/topic challenge innovation_替换为真实ID --round 1
uv run academic-agent --project projects/topic assess-innovation innovation_替换为真实ID
uv run academic-agent --project projects/topic innovation-status
```

完成逐篇阅读和复核后，可选择明确作品、领域包和已验证依赖生成跨文献大纲与正文。命令返回的 `output` 是具体 `v-*` 目录；后续命令应使用该目录，不能用 `latest`：

```powershell
uv run academic-agent --project projects/topic domain-extract --question "研究问题" --work-id work_A --work-id work_B --domain builtin:scheduling
uv run academic-agent --project projects/topic writing-outline --question "研究问题" --work-id work_A --work-id work_B --domain-version "实际 domain output"
uv run academic-agent --project projects/topic writing-draft "实际 outline output"
uv run academic-agent --project projects/topic writing-export-edit "实际 writing output" --output human-edit.json
uv run academic-agent --project projects/topic writing-import "实际 parent output" human-edit.json --author "研究者" --reason "修改原因"
uv run academic-agent --project projects/topic writing-review "实际 pending output"
uv run academic-agent --project projects/topic writing-status
```

外部领域包是只读 JSON 数据，必须使用绝对路径；它能改变提取字段、比较列和写作结构，但不能绕过来源、复核、预算或产物完整性检查。人工导入先保存原文件 bytes/hash，再对变化文字划分并独立复核；旧版本保持可读，来源、筛选、人工复核或领域包变化会显示为过期并给出恢复命令。

`challenge` 得到的新书目仍需逐篇 `screen`；要作为反例证据进入评估，还需用 `attach` 提供可读 PDF、Markdown 或 TXT，再执行 `read` 和 `review`。每个候选最多两轮，每轮最多两条定向查询；重复同一轮会恢复已有进度。评估完成只表示四维结构与逐单元复核已写入，`not_searched`、来源失败、证据不足和待人工学术判断仍会原样保留；它不证明全球首次或研究质量。

完整的离线与真实使用步骤见 [创新候选使用指南](docs/innovation-walkthrough.md)。

`ask` 按论文回答并复核，返回陈述、证据和 `usable` 状态。手动复核使用 `review --claim-id ... --decision supported --reason ... --reviewer 姓名`，保留身份和版本。所有命令支持 `--help`，错误返回 JSON 和非零退出码。

任何阶段都可运行 `guide` 获取按当前状态生成的下一步。遇到待筛选记录时，它只给出逐篇命令；缺少可读全文、需 OCR、阅读卡失效、尚未复核、预算用尽、exchange 待人工响应、导出或趋势过期时都会分别说明。趋势报告的计数单位是库内去重书目记录，不等同独立研究；本地导入覆盖未知，当前年未结束，小样本只作描述。

## 配置与后端

每个项目有 `project.json`，可修改来源、查询、模型、代理和预算。默认 100 篇候选、20 篇阅读材料、1 个模型工作者，最多 3 个并发模型工作者。读与审采用独立调用，通常每篇至少 2 次。

- **codex**：默认复用 Codex 登录，在临时目录中启动隔离、只读、禁用研究模型工具的子进程，不传入独立 API key。限量合成材料观察已完成领域提取、领域复核、正文及独立全文复核；原文数字、条件和引用已检查，证据不足决定原样保留。历史失败及每轮授权范围见验收记录；模型结果仍需研究者检查。
- **demo**：只接受带 `ORIGINAL_SYNTHETIC_FIXTURE` 标记和规定标签的合成材料，不提供真实论文分析。
- **exchange**：在项目 `exchange/` 写出请求；放入符合 Schema、包含相同 fingerprint 和 data 的 `.response.json`，再重试/恢复，不自动调用外部服务。

默认国外请求经 `http://127.0.0.1:7897`，国内/本地直连。可用 `init --proxy` 或配置文件修改；此版本对国外服务要求显式代理。下载拒绝私网目标及不安全跳转；DNS 未固定到最终连接，系统与代理 DNS 仍需可信。不绕过全文访问限制。

`--offline` 禁用**论文来源和出版状态查询**并标记未知，Codex 模型仍需要网络。完全无网络演示使用 `demo`。

预算以“后端尝试次数”计量，包含超时或结果未知的尝试；demo/exchange 也经过同一路径，usage 表明未调用模型。未知货币费用输出 `null`。预算用尽后修改配置再恢复，输入相同的成功任务会复用。`cancel JOB_ID` 阻止结果写入，不能保证撤销提供方已接收的请求。

## 输出与资料

`exports/latest.json` 指向新版本目录：

- `review.md`、`reading-cards.md`、`comparison.md/.csv`、`outline.md`、`gaps.md`
- `library.json/.csv`、`references.bib`、`evidence-ledger.json`、`query-log.json`
- `quality.json`、`manifest.json`

导出前重新检查出版通知；已知撤稿不能支持普通结论，无法查询保持未知。每次创建新目录，不覆盖人工稿件。原文变更使旧阅读卡失效；`status` 显示导出是否过期。研究数据、PDF、数据库、凭据和本地运行记录均不提交 Git；数据库包含全文证据，应作为个人研究材料管理。

## 研发与验收

[开发计划](docs/development-plan.md) · [开发记录](docs/development-ledger.md) · [验证结果](docs/validation.md) · [0.3 验收记录](docs/validation-v0.3.md) · [0.4 验收记录](docs/validation-v0.4.md) · [0.4 本地发行范围](docs/release-0.4.0.md) · [架构](docs/architecture.md) · [扩展](docs/extending.md) · [Git Flow](docs/git-flow.md) · [参考来源](docs/research-sources.md)

先创建 `.runs` 目录，然后：

```powershell
uv run pytest --basetemp .runs/pytest-local
uv run ruff check .
uv build
```

CI 配置 Windows/Linux 与 Python 3.11/3.12/3.14 离线验证，实际结果以仓库运行记录为准。真实题目的召回率、人工证据准确率与研究效用尚未验证，单元测试数量不能代替这些评价。

安装后还可在一个全新目录运行完整离线使用自检：

```powershell
uv run academic-agent self-test --output .runs/usage-local
```

自检通过当前 Python 调用实际包入口，覆盖中文空格路径、RIS、显式筛选、全文、预算恢复、版本保留、过期检测、趋势重算、错误批注、可读/无文字 PDF，以及一个原创候选到四维评估的缓存恢复与过期旅程。它不依赖 pytest，不访问真实模型或论文来源，并在 `usage-report.json` 与 `usage-report.md` 中记录每条 argv、退出码、耗时和实际产物断言。输出目录必须尚不存在；报告是操作验收证据，不是人工学术准确率评测，也不验证真实来源召回率或全球新颖性。
