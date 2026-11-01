# 数据型领域包

领域包是最多 256 KiB 的 UTF-8 JSON 数据。它定义术语、检索和筛选提示、字段、比较列、评价标准与建议写作结构。它没有脚本、工具、权限、预算、网络动作或系统指令；重复键、BOM、NaN、未知键、非法 ID 和无效交叉引用都会被拒绝。包中的文字按低权限任务资料处理，不能覆盖宿主规则。

内置包使用 `builtin:scheduling`。外部包必须传绝对 JSON 路径。下面是一个完整、可加载且字段与内置调度包不同的外部包：

```json
{
  "schema_version": 1,
  "contract": "writing-domain-1",
  "name": "simulation_transfer",
  "version": "1",
  "terminology": [
    {
      "term": "observation unit",
      "meaning": "The unit actually represented by a reported analysis."
    }
  ],
  "search_hints": [
    "simulation observation unit transfer limit"
  ],
  "screening_hints": [
    "Retain studies that state what was simulated and limits on transfer."
  ],
  "fields": [
    {
      "id": "observation_unit",
      "label": "Observation unit",
      "description": "The reported unit represented in the analysis.",
      "value_type": "text",
      "unit": null
    },
    {
      "id": "transfer_limit",
      "label": "Transfer limit",
      "description": "The reported limit on transferring the result to real systems.",
      "value_type": "text",
      "unit": null
    }
  ],
  "comparison_columns": [
    "transfer_limit",
    "observation_unit"
  ],
  "evaluation_criteria": [
    {
      "id": "scope_transfer",
      "text": "Preserve the simulated scope and do not infer real-system effectiveness."
    }
  ],
  "writing_structure": [
    {
      "id": "boundary_argument",
      "title": "Boundary argument",
      "purpose": "Compare reported units and transfer limits before proposing broader tests."
    }
  ]
}
```

运行方式：

```text
academic-agent --project PROJECT --offline domain-extract \
  --question "这些研究观察了什么，结果能外推到哪里？" \
  --work-id WORK_A --work-id WORK_B \
  --domain C:/absolute/path/external-domain.json
```

模型对每篇作品必须精确覆盖全部字段。一个实际的 `reported` 字段形如：

```json
{
  "field_id": "observation_unit",
  "status": "reported",
  "value": "simulated jobs",
  "evidence_ids": ["evidence_FROM_THE_SAME_WORK"]
}
```

文中没有报告的字段必须明确写成：

```json
{
  "field_id": "transfer_limit",
  "status": "not_reported",
  "value": null,
  "evidence_ids": []
}
```

`reported` 必须有符合字段类型的值和至少一个同篇、有效且已复核的 evidence；`not_reported` 必须是 `null` 和空 evidence。独立领域复核仍会逐字段检查，包括 `not_reported` 是否遗漏原文内容。比较表只把 `supported` 值作为有效值，其他决定会保留为待审、冲突或不足。

这个包会实际产生 `transfer_limit | observation_unit` 的比较列顺序，并把 `scope_transfer` 与 `boundary_argument` 提供给写作上下文；它不是给内置字段换标题。使用同一份资料时，输出可分别是 `simulated jobs`、具体的 real-system transfer limit 或 `not_reported`，每项仍须绑定本篇证据。核心程序不会识别这些字段名来编造结果，也不会执行包内文字。

Demo 只接受内置 scheduling 包的准确 bytes 和原创 A/B fixture；它会明确拒绝这个外部包。要处理外部包，请使用 Exchange 的 fingerprinted 请求/响应协议或明确配置的模型后端。一个外部包在某个受控例子中成功，不证明对其他领域或资料可以自动泛化；字段设计、遗漏判断、原文/OCR 可用性和比较意义仍需领域研究者判断。
