# 写作阶段接口预核对

2026-09-09。为 0.4 草案核对已稳定的接口；**不是已实现功能或最终实施计划**。下列候选、反例和评估接口已完成各自的 0.3 组件审查，写作适配仍待实施。

| 已核对接口 | 写作实现应遵循的约束 |
|---|---|
| `ResearchSnapshot`、`freeze_research`、`read_research`、`research_fingerprint` | 复用当前原文、证据、阅读卡和有效决定的共同校验，保留完整所选原文；只读状态不能创建 Store。自己的模型计量和进度不进入资料身份。 |
| `publish_research`、`load_research` | 当前允许的种类只有 innovation/challenge/assessment。写作或领域提取产物需要明确扩展种类集合和严格 manifest 类型，并测试旧种类兼容，不能把写作伪装成 assessment 来绕过契约。 |
| `verify_trends(directory)` | 纯文件校验并重算旧六文件产物，返回有效性、算法/输入身份、源快照和校验数；不返回完整统计对象。调用者不能假设返回值已有 `computation`。 |
| `CorpusSnapshot`、`ComputationRecord`、`compute_trends` | 已有冻结语料和确定性计算契约。需要从明确选择且校验过的文件取得完整对象，保留算法、口径、分母与输入身份；来源原话和语料派生值是不同证据类型。 |
| `CorpusSnapshot.source_snapshot` | 含旧导出计量/日志相关身份。选定的历史统计描述冻结语料，不能每次写作后重新比较这个运行计量元组，就把没变的统计依据标成失效。 |
| `MeteredBackend.validated()` | 领域提取、大纲、正文和复核各自完成领域校验后保存成功结果，暂停后复用已经成功的阶段；无效结构不能提前进入成功缓存。 |
| `load_candidate(root, candidate_id)` | 已通过 0.3 组件审查的只读历史入口，返回候选、批次、依据审查、原快照和版本来源。写作可引用历史研究设想，但不能将其当作已经完成的实验结果。 |
| `candidate_currentness(root, candidate_id)` | 复用原资料前提有效性政策，返回 current 与分类缺口；历史文件完整不等于原依据现在仍然有效。新增反例不强迫更换候选身份。 |
| `read_challenges(root, candidate_id)` | 返回已开始的各轮、实际查询尝试、当前命中状态及待办。查询待办带可选 round；全文/OCR 访问级别独立于筛选资格。写作应保留每轮缺口，不能用旧轮成功消去新轮未检索。 |
| `innovation_status(root)` | 只读返回各候选的历史可用性、`premises_current` 和评估的 `available/current/output/status/gaps/input_hash`。评估状态为 missing/current/stale/invalid；当前缺口报告可以同时有 `assessment.current=true`、`premises_current=false`，不能将前者单独解释为依据有效。 |
| `assess_candidate(runner, candidate_id)` | 这是执行评估的有状态入口，不能用于写作状态查询。结果 complete 仍可包含未检索或证据不足；它不接受 `work_ids`，不应由写作适配器假定存在该参数。 |

评估版本包含 `candidate.json`、`assessment.json`、`assessment-review.json`、`context.json`、`report.md`、`references.bib`、`evidence.json` 及宿主 manifest。`load_research` 校验文件完整性，`innovation_status` 提供当前状态；二者的作用不同。后续计划需规定如何固定同一版本的已校验字节、结构化对象与当前状态，不能在状态检查后随意读取另一个 latest 版本。

评估的四个维度为 `novelty_clues`、`research_value`、`feasibility`、`evidence_strength`。`evidence_sufficiency` 的 insufficient/available_within_scope 描述资料可用性，不是创新性或学术质量评分。独立复核逐个覆盖维度判断、理由、缺口及最低验证方案文字；写作消费时不能省略冲突或证据不足的决定。分析输出版本或 UUID 本身不改变语义身份，但原文 `source_path` 仍是追溯依赖，不能额外承诺原文迁移不使稿件过期。

统计读取还需解决明确的文件变化窗口：校验与随后读取之间若文件被改写，不能把另一份内容当成刚刚通过的结果。实施计划应规定所保存字节与已校验哈希的核对，以及提交前的依赖检查；“先运行校验命令，再随意读 JSON”不足以建立同一份证据。

正文由有序文本单元构成的选择，可使宿主精确检查文字覆盖。单元的“连接语”或“假设”标签仍需独立语义检查，不能仅凭标签豁免事实审查。人工修改导入后的文本应保持原样进入划分和复核，不能由模型先删除错误再声称原修订通过。

历史候选与多轮反例入口已由 0.3 任务 3 实现并审查通过（修复提交 `67a31d1`），评估与当前状态由任务 4 实现并通过审查（源提交 `d793d3e`，路径契约澄清 `35e0e41`）。写作消费的完整检查窗口、领域字段值与文档证据绑定仍需在后续实施计划中固定；不能提前宣称写作或整版兼容验收已完成。
