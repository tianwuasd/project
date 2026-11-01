# 写作、人工修订与独立复核

写作功能把已复核研究材料、一个明确的领域提取版本以及可选的统计、候选和评估版本固定在同一份上下文中。它生成的是可追溯的研究工作稿，不代表实验已经开展、结论已经获得人工确认或稿件已经达到投稿标准。

## 从领域提取到正文

下面的路径都应替换为命令实际返回的 `output`，不要使用 `latest`，也不要猜版本 UUID。

```text
academic-agent --project PROJECT --offline domain-extract \
  --question "不同负载下结果如何变化？" \
  --work-id WORK_A --work-id WORK_B \
  --domain builtin:scheduling

academic-agent --project PROJECT --offline writing-outline \
  --question "不同负载下结果如何变化？" \
  --work-id WORK_A --work-id WORK_B \
  --domain-version PROJECT/analyses/domain/v-ACTUAL

academic-agent --project PROJECT --offline writing-draft \
  PROJECT/analyses/writing/v-ACTUAL_OUTLINE

academic-agent --project PROJECT --offline writing-status
```

`writing-outline` 也可显式固定 `--trends DIR`、`--candidate-version DIR --candidate-id ID` 和 `--assessment-version DIR`。候选目录和 ID 必须同时提供，评估必须属于该候选。命令会核对领域版本保存的问题和有序 `work-id`，并重新捕获这些可选版本；不会丢弃选择、换成最新版本或修改领域产物。

领域版本位于 `PROJECT/analyses/domain/v-<32hex>/`，其中有 `domain.json`、`domain-review.json`、`comparison.md`、`references.bib`、`evidence.json` 和 `context.json`。写作版本位于 `PROJECT/analyses/writing/v-<32hex>/`，其中有 `outline.json`、`draft.json`、`review.json`、`revision.json`、`raw-edit.bin`、`report.md`、`references.bib`、`evidence.json` 和 `context.json`。`report.md` 是由结构化文本单元和复核决定渲染的工作稿；每个来源单元的审计脚注显示 citekey、原始 document 路径及 claim/evidence 定位，跨文献综合列出每项来源，冻结统计另列 computation、trend pin 和可用的来源路径。路径由宿主从已校验引用生成并安全转义，不改写正文单元。若合法来源只有捕获的摘要而没有独立源文件，脚注会明确标为 `abstract-only`、显示 document ID，并链接同版本的不可变 `context.json`；该文件保存精确 document/pages，而不会把空路径误链接到项目根。不要直接修改正式版本，哈希校验会拒绝它。

命令返回的顶层 `status=complete` 表示这次操作完成。`support_review` 单独报告逐单元支持情况；其中出现 `partial`、`contradicted` 或 `insufficient` 不会把已完成操作伪装成失败，但这些缺口必须由研究者处理。预算暂停、等待 Exchange 输入、输入过大、过期和失败均以非零退出，并保留 `gaps`、解释和可执行的恢复动作；程序不会自动增加预算或截断原文。

## 精确的可编辑 JSON

先导出一个明确版本；目标文件以 exclusive-create 创建，已有文件不会被覆盖。

```text
academic-agent --project PROJECT --offline writing-export-edit \
  PROJECT/analyses/writing/v-PARENT \
  --output PROJECT/edits/revision-1.json
```

文件的完整顶层结构如下；每个块也只能有这里列出的四个键。

```json
{
  "schema_version": 1,
  "document_id": "doc_0123456789abcdef0123456789abcdef",
  "parent_version": "v-0123456789abcdef0123456789abcdef",
  "parent_manifest_hash": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "outline": [
    {
      "id": "b_11111111111111111111111111111111",
      "role": "heading",
      "level": 1,
      "text": "条件限定的调度证据"
    },
    {
      "id": "new:outline-limit",
      "role": "paragraph",
      "level": null,
      "text": "明确比较负载条件与外推边界。"
    }
  ],
  "paragraphs": [
    {
      "id": "b_22222222222222222222222222222222",
      "role": "heading",
      "level": 2,
      "text": "反例与局限"
    },
    {
      "id": "new:revised-paragraph",
      "role": "paragraph",
      "level": null,
      "text": "A 的平均等待时间为 10→8；B 在突发到达下为 10→11。\r\n这两项模拟不能证明真实系统效果。"
    }
  ]
}
```

保留的块继续使用父版本 ID。新增块使用唯一的 `new:<slug>`；导入后宿主会分配稳定 ID，并在 `revision.json.id_map` 保存映射。删除块时从相应数组移除，并对每个缺失的父块另加一个 `--delete-id ID`；清单必须恰好等于所有被删 ID。数组顺序就是最终顺序，因此调序只需改变数组顺序。原 ID 不能从 `outline` 搬到 `paragraphs`，反之亦然。`heading` 的 `level` 为 1–3，`paragraph` 的 `level` 必须是 `null`。不允许空白块或空文稿。

JSON 源文件行尾可为 LF 或 CRLF。字符串里的 `\r\n` 是 JSON 编码，解析后成为正文中的两个真实码点；直接在 JSON 引号内放未经转义的换行不是合法 JSON。导入不会正规化空格、组合字符、非 BMP 字符或正文 CRLF。文件起始可有一个 UTF-8 BOM；`raw-edit.bin` 保存包含 BOM 和外部行尾在内的完整原始 bytes，`input_sha256` 对这些 bytes 计算。划分偏移使用 Python Unicode code point 的半开区间，不是 UTF-8 bytes 或 UTF-16 code units；模型只能返回覆盖原文的 span 和分类，不能返回替换文本。

导入命令显式记录作者、理由和删除确认，不调用模型：

```text
academic-agent --project PROJECT --offline writing-import \
  PROJECT/analyses/writing/v-PARENT PROJECT/edits/revision-1.json \
  --author "Researcher Zhang" \
  --reason "Correct the workload condition and add the transfer limit." \
  --delete-id b_DELETED
```

内容变化、调序或依赖语义变化会产生 `stage=pending`，并清空当前整稿复核。若再次导入一个尚未复核的 pending 父版本，新版本仍为 pending，所有仍存在的未完成划分目标都会保留；即使第二次只改变文件格式，或纯调序而没有划分目标，也不能绕过一次完整整稿复核。随后对命令实际返回的 pending `output` 运行：

```text
academic-agent --project PROJECT --offline writing-review \
  PROJECT/analyses/writing/v-ACTUAL_PENDING
```

`extent` 表示内容范围，`stage` 表示复核阶段，两者互不替代：

| extent | stage | 含义 | 下一步 |
|---|---|---|---|
| `outline` | `pending` | 人工大纲，正文为空，尚待完整复核 | `writing-review` |
| `outline` | `reviewed` | 完整大纲已执行独立复核，正文仍为空 | `writing-draft` |
| `manuscript` | `pending` | 大纲和正文都存在，当前整稿尚待复核 | `writing-review` |
| `manuscript` | `reviewed` | 大纲和正文都存在，完整逐单元复核已执行 | 处理支持缺口并继续修订 |

因此，可导出一个 outline 版本，保持 `paragraphs: []` 修改大纲，导入得到 pending outline，再运行 `writing-review`，最后把实际 reviewed outline 输出交给 `writing-draft`。人工编辑不能借此把 outline 直接变成 manuscript。

## 来源、复核或领域包变化后的恢复

当人工作者稿因原文、有效复核或领域包变化而过期时，先保持父版本、编辑文件 bytes、作者、理由和删除清单不变。第一步只重新做基础领域提取：

```text
academic-agent --project PROJECT --offline domain-extract \
  --question "SAVED QUESTION" \
  --work-id SAVED_WORK_A --work-id SAVED_WORK_B \
  --domain SAVED_PACK_SOURCE
```

这一步没有趋势、候选或评估开关；后续重新绑定会按父稿的原始明确选择重新捕获它们。领域提取完成前，`guide` 和失败结果只显示这条可执行动作，并说明下一步，不会制造不存在的版本路径。取得它实际返回的 `output` 后，再运行：

```text
academic-agent --project PROJECT --offline writing-import \
  SAME_PARENT SAME_UNCHANGED_EDIT_FILE \
  --author "SAME AUTHOR" --reason "SAME REASON" \
  --delete-id SAME_DELETED_ID \
  --domain-version PROJECT/analyses/domain/v-ACTUAL_FRESH
```

最后，对该导入实际返回的 pending `output` 运行 `writing-review`。新版本的 `revision.json.domain_rebind` 保存 old/new 领域 pin，父版本和旧领域产物保持不变，人工 text 不会被模型改写。重绑会比较完整的写作语义上下文与领域语义；例如候选的反例覆盖从未读变为可读，即使基础领域 hash 不变，也会清空旧支持、划分和整稿复核并要求重新检查。只变 UUID、日志或计量且语义等价时仍可复用。

`evidence.json.block_provenance` 按块保留最初的人类作者或模型来源、时间和父链。精确未变的 text/role/level 即使因标题、领域或上下文变化而必须重新划分，也保留经父 manifest 核验的原内容作者；真正修改或新增的内容才记录本次人工作者。后来的 `review_provenance` 是独立复核者，不会被错误写成原文作者。`reused_from` 可说明内容或划分来源，不代表当前支持决定仍然有效。

## 证据边界

`frozen_statistic` 表述只描述所选统计版本冻结的语料，固定其值、口径、分母、时间、算法和输入哈希。它不是当前总体的新统计，也不会自动换成最新统计。标题、连接语和每个实质句都进入独立复核；有引用并不自动表示语义获支持。

`abstract_only`、`needs_ocr`、无法访问的全文以及检索覆盖缺口会限制可写内容。系统不会用摘要补成全文结论，也不会把 OCR 失败、模拟任务、候选假设或拟议实验写成真实参与者和已观察结果。研究者仍须检查原文访问条件、OCR、数值与否定、引用归属、领域适用性和学术表达。

Demo 只验证内置领域包准确 bytes 与两份原创合成材料的受控协议，它不衡量一般模型准确率。Exchange 保存带完整 fingerprint 的请求/响应，可用于明确的人工或受控后端；真实模型观察、结构校验结果和最终人工学术判断应分别记录，不能把预设 Demo/Exchange 响应报告为准确率。
