# 当前证据快照与分析版本

本模块是 0.3 的公共存储层，不调用模型、不搜索，也不判断论文的学术正确性。当前支持三个产物种类：`innovation`、`challenge`、`assessment`。

## 快照接口

```python
snapshot = freeze_research(store, work_ids=None, require_supported=True)
snapshot = read_research(project_root, work_ids=None, require_supported=True)
fingerprint = research_fingerprint(snapshot)
assert_research_current(store, snapshot)
payload = snapshot.stable_payload()
```

`ResearchSnapshot` 禁止未知字段并验证工作—文档—证据—陈述—决定的完整依赖链。字段如下：

| 字段 | 含义 |
|---|---|
| `schema_version`, `captured_at` | 协议版本 1 与本次捕获时间 |
| `selection`, `require_supported` | 默认或显式选择，以及是否要求每篇至少一个已支持陈述 |
| `work_ids`, `works` | 按 ID 排序的所选文献列表、以 Work ID 为键的完整元数据 |
| `documents`, `evidence` | 分别以 Document ID、Evidence ID 为键；包含全部可读页面原文及精确证据，不截断 |
| `cards`, `decisions` | 分别以 Work ID、Claim ID 为键；阅读卡内保留 Claim；决定包含文献、文档、卡片哈希、证据 ID、理由、复核人及 model/manual 类型 |
| `question`, `scope` | 当前问题、项目标题、配置检索来源与查询 |
| `coverage` | 全项目文献总数、纳入、待筛选、排除、撤稿、此次选择数量；撤稿数与筛选分类可重叠 |
| `search_events` | 原检索事件的 kind/data/created/id，保留查询、来源响应、失败、截断和有效上限等既有信息 |
| `gaps` | 文献级或项目级缺口；`work_id=None` 表示项目级 |

`supported_claim_ids` 是按 ID 排序的属性，仅返回有效决定为 `supported` 的陈述。未复核、部分支持和证据不足的陈述仍在阅读卡中，下游不得把它们当作已支持陈述。

默认仅选择已纳入且未撤稿的文献。显式 ID 必须存在、唯一、非空并符合相同资格条件；顺序会规范化。默认严格模式拒绝空语料、不可读/过期文档，以及任何没有至少一个已支持陈述的所选文献。`require_supported=False` 可返回空的默认选择并保留待阅读、待复核条目，但文档和证据只暴露通过验证的可读内容。过期的元数据摘要记为缺口，不会被当作当前原文。损坏数据会报错，不通过退回较早的阅读卡或复核结果隐藏当前无效记录。

缺口代码包括 `missing_document`、`stale_source_abstract`、`unreadable_document`、`abstract_only`、`missing_current_card`、`missing_supported_claim`、`pending_review`、`not_searched`、`search_incomplete`、`search_truncated`、`project_cap_reached`。具体失败与查询范围应同时读取 `search_events`；没有命中不能等同没有反例。未选文献的筛选/撤稿状态通过全项目覆盖数量可见。

## 只读与指纹

两个读取入口共用同一投影与校验。`read_research` 不构造 Store，要求已有 `project.json` 和 `research.db`；缺失或仅有配置时直接报错，不建立目录、数据库或 blob。已有数据库使用 `Path.resolve().as_uri()` 编码后追加 `?mode=ro`，执行普通读取事务，能看到已提交的 WAL 数据，支持目录中的字面 `#`、`%`。不使用 `immutable=1`。SQLite 自身可能建立或维护选定数据库的 WAL/SHM 协调文件；不写权威研究行、配置或 blob。

捕获前后分别读事务投影并比较有意义的指纹；读取过程中源数据改变时拒绝结果。文档 JSON blob 与原始附件副本分别检查哈希、路径和证据跨度。Store 保存的附件副本是权威原文；外部导入路径随后被修改，不会自动替换已经保存的副本，必须重新 attach。

指纹固定所选元数据、原文内容及 ID、所有有效证据、当前阅读卡和有效决定、问题/范围、全项目覆盖数量与检索事件。排除快照捕获时间、文档创建时间、Work 的操作性 revision、模型运行/计量、分析日志、预算及代理配置；**不使用 `exports.snapshot_token`**。检索事件自身的时间、结果和上限属于检索范围，予以保留；已有出版检查时间也保留为元数据。

`snapshot.stable_payload()` 提供稳定的依赖与输入基础，完整快照仍保留原始检索事件。后续模型任务按 Task 2 声明的检索覆盖投影构造模型输入：保留查询范围、覆盖状态与事件哈希，不直接发送全部原始检索响应或未选文献元数据；所选文献的全部可读原文及证据—陈述—决定链必须完整保留，超过容量时拒绝，不能截断。不要直接将包含 `captured_at` 的 `model_dump()` 用作缓存输入。此接口排除的文档创建时间仍存在历史 manifest 中供审计。再次读取与复用成功模型结果，不会仅因新增日志或捕获时间而改变任务身份。

## 不可变版本与读取契约

```python
directory = publish_research(store, "innovation", snapshot, {"candidate.json": candidate_bytes})
record = load_research(directory)
latest = load_research(project_root / "analyses" / "innovation")
```

目录格式固定为 `analyses/<kind>/v-<32位小写十六进制UUID>/`。版本由宿主产生，模型不能指定。每个版本含 `manifest.json` 与调用者提供的精确 payload 文件集；不硬编码后续候选/评估的模型协议。文件名只允许 1–120 位 ASCII 字母、数字、点、下划线和连字符，以字母或数字开头；拒绝路径、Windows 保留名、流路径、末尾点、大小写碰撞，以及 `manifest.json` / `latest.json`。文件必须为 bytes。

Manifest 的精确字段为 `schema_version`、`kind`、`version`、`snapshot`、`fingerprint`、`files`、`manifest_hash`。`files` 是文件名→SHA-256 映射。JSON 使用 UTF-8、排序键、两空格缩进、不附加换行；`manifest_hash` 对去除自身后的同一规范 JSON 计算 SHA-256。最新指针为同目录 `latest.json`，仅含 `schema_version`、`version`、`manifest_hash`，同样使用规范 JSON。缺少、额外、损坏、非规范文件及符号链接/目录联接均拒绝。

`load_research` **精确返回** `{"manifest": dict, "files": dict[str, bytes]}`。传入版本目录读取历史；传入种类目录验证最新指针后读取目标。不接受自由路径或将最新指针中的相对路径直接拼接访问。仅加载历史时不要求活项目仍与历史相同；是否过期由调用者验证存储快照的当前性。

发布前检查输入，写新的 `.pending-*` 暂存目录，再次检查输入；取得 `BEGIN IMMEDIATE` 后，在同一数据库连接上重新校验，检查历史产物，移动为新版本，并在更新最新指针前重新校验文件和当前输入。指针通过 `atomic_json` 原子替换。每次发布保留旧版本；历史中的人工修改或损坏会报告错误，不替换或修复。失败暂存目录、已移动但尚未成为 latest 的版本保留，便于调查。

哈希提供修改检测，**不证明作者身份**；有能力同时重写内容与全部哈希的人仍能伪造文件。SQLite 锁只保护遵守数据库事务的写入，不能锁住外部配置、blob 或分析文件；前后重复读取可发现观测到的修改，不能承诺跨 SQLite 与文件系统的单一事务，也不防止恶意进程在最后检查后修改文件。校验失败不会覆盖之前的 latest；进程在版本移动后停止可能留下可验证的未引用历史版本。目录本身不设为操作系统只读，用户可以编辑副本，改动将被完整性检查报告。
