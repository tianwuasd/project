# 0.4.0 写作与领域扩展验证记录

日期：2026-09-13，发行前冻结。离线安装使用、已有代码复审及所需四阶段真实 Codex 观察均已完成。实际整稿严格加载、来源哈希、引用证据及逐单元复核覆盖通过；13 supported/1 insufficient 均保留。真实观察使用两篇合成材料，经历明确授权的有界恢复，不是一次不中断的全真实链路，也不构成学术准确率评测。历史失败和定向验证分别保留。最终 docs-bearing 包、Git Flow 推送及正式提交六环境 CI 由 checkout 外发行收据记录，完成前不宣称已发布。

## 已执行范围

Task6 从 `b1edc1c624542da3a52e11694488c6f83d25cd34` 开始。最终 Task6 影响范围运行 `python -m pytest tests/test_writing_usage.py tests/test_usage.py tests/test_backends.py -q`，结果为 **64 passed in 270.39s**；六个改动源文件/测试文件的 Ruff 检查通过。Task6 修正轮已通过独立复核。控制器随后按计划运行一次整分支 pytest 和 Ruff；pytest 的实际 1/619 结果及定向修正见下文，不能提前或事后改写成全量通过。

Task6 第一次独立复审指出四项本地证据缺口。修正轮的 focused oracle 为 **5 passed in 0.41s**；源代码集成自检为 **1 passed in 285.40s**。第一次集成 RED 保留了 `writing-status` 的完整树差异和长路径 clone 固定预算耗尽；Ruling24 将 SQLite 只读锁 sidecar 与应用状态写入分开，并批准独立的固定 60/500000 长路径 fixture。修正轮没有改写生产工作流、写作 producer、数据库或预算默认值。

控制器在干净 `c9f30237a0e3725ddce2059a225b55e003d07a79` 上的第一次整分支 pytest 为 **1 failed, 619 passed in 736.09s**、exit 1。唯一失败 `tests/test_domain_analysis.py::test_input_too_large_and_changed_source_statuses_are_precise` 仍期待只替换完整 source blob 后返回 `failed`，实际生产结果是 `stale`。Task3 Ruling12 明确规定：其余研究状态验证通过时，source hash 单独变化是 `stale`；source 变化与 malformed/missing/unsafe 状态并存才是 `failed`。Task6 fix round2 只改测试和状态文档：更名并把 source-only 期望改为 `stale`，再加入 source drift + malformed review 的 `failed`/零 backend-job 负例。`tests/test_domain_analysis.py` 加四个 `research_state_currentness` 边界和一个 writing-context 边界共 **17 passed in 14.84s**，touched Ruff 通过。生产 runtime 未改，也没有重跑全套或安装旅程，因此没有 `620 passed` 声明。

隔离候选包由 `uv --cache-dir <local-workspace>/.tool-cache build --offline` 构建。缓存中的隔离构建后端只用于构建，不能描述为锁定运行时。运行依赖由 `uv export --locked --offline --no-dev --no-emit-project` 导出并带哈希，在 checkout 之外的 `<local-workspace>/.runs/academic-agent-v04-release-20260912-01/runtime` 中离线安装 42 个依赖，再以 `--no-deps` 安装候选 wheel。该候选 wheel 为 195392 bytes、SHA-256 `df28c09023d370fd173c3c725f3b52040ffae41e72fac218d313ec7cf77b02b1`；候选 sdist 为 509608 bytes、SHA-256 `6055a14d4af7b889d9d7e32f32963ddb860a83e2609949da62b76c0ba2b59216`。发行文档写入后的最终待发布包须另行构建和核验，不能把这两个候选哈希写成已发布哈希。

从 checkout 外以安装环境的 `python -I -m academic_agent.cli self-test` 运行，实际报告是 `<local-workspace>/.runs/academic-agent-v04-release-20260912-01/installed-self-test/usage-report.json`，942800 bytes、SHA-256 `f907da3f80332a252b3f96304cc90ab85f7a158b0374fc96e6bad8862bf3602f`。结果为 **17 个场景、172 条命令、249 项断言全部通过**，实际模型/论文网络调用为 0。安装来源位于该运行环境的 `Lib/site-packages`，版本为 0.4.0，pytest 不可用；34 个 Python 模块及内置领域包与受测源文件逐 bytes 相同。候选 sdist 已核对 `uv.lock`、0.4 规格/设计、写作用例和固定趋势夹具；最终 sdist 还必须包含本验收记录及发行记录。

修正轮从新 wheel 建立了 checkout 外的新 Python 3.14.3 runtime：`<local-workspace>/.runs/academic-agent-v04-fix-round1-20260912-01`。安装报告 `installed-self-test/usage-report.json` 为 1842173 bytes、SHA-256 `55e383ef547a86d771277c62b52617f1fda048099e10b6ad389afc6eae4c579d`，结果为 **17 个场景、184 条命令、268 项断言全部通过**，实际模型/论文网络调用仍为 0。新 wheel 为 198973 bytes、SHA-256 `182947705ebfab45ab35cdcd4e49dceca90a727d7625de157e00c4b862c60171`；`package-byte-verification.json` 证明 35 个 payload 文件在受测 source、wheel 和安装目录逐 bytes 相同，pytest 不可用。锁定导出含 43 个 requirement entries 与 852 个 SHA-256 flags，安装了当前平台适用的 42 个运行依赖；wheel generator 为 `setuptools (84.0.0)`。该修正 wheel 仍是本地候选，控制器最终文档包重建门槛不变。

最有能力模型首次整分支复审在 `8e67e2a8a3a934a81cdba850dd732aede32b3f11` 上提出四项 Important：把内容作者与重新划分/支持分开；第二次导入继续携带 pending 状态和目标；显式重捕获时比较完整写作语义上下文；在 Markdown 单元脚注显示宿主生成的来源定位。初始 focused RED 的 7 个失败中只有 6 个准确到达 I1/I2/I4 边界；原 I3 用例先在依赖 currentness 停止，未冒充目标 RED。修正后的真实 I3 状态迁移先保存 hit/search/membership/screen/source，再仅把反例从 unread 读审为 ready；fresh base-domain 语义 hash 相同而完整写作上下文不同，旧实现错误保留 reviewed。该单项 RED 为 1 failed/9.14s。实现后的 7 项 focused 为 **7 passed in 29.87s**，三文件覆盖 `test_writing_revisions.py`、`test_writing_workflow.py`、`test_writing_usage.py` 为 **55 passed in 127.66s**。

第一次新的完整源自检保存为 **1 failed in 317.80s**：新增统计可见引用 proof 在主 fixture 消耗两次受控调用，使后续 stale-rebind 场景到达固定 60 调用预算并暂停。修正只把该 proof 移入一个 byte-identical、独立初始 60/500000 的受控 Exchange fixture，没有提高或重置原预算；写作五场景随后为 **152 条命令、210 项断言通过**。新的候选 wheel 为 203062 bytes、SHA-256 `cb64908d4a83b22860f4581d2d4717bb3c68095af36e0520fd621be83d1ae04f`。在 checkout 外 `<local-workspace>/.runs/academic-agent-v04-final-fix1-installed-20260912-01/runtime` 安装锁定哈希依赖及该 wheel 后，完整安装报告 `installed-self-test/usage-report.json` 为 **2047976 bytes**、SHA-256 `285086f21d267fc583698aa43123c88dcbc6df26be8f7f61f13a88c066113374`，**17 个 canonical 场景、209 条命令、299 项场景级加命令级断言全部通过**；版本为 0.4.0、来源为 outside site-packages、pytest 不可用、实际模型/论文网络调用为 0。第一次 wheel-install 包装命令误用 PowerShell `Select-Object -Single`，在启动 uv 前 exit 2；原收据保留，正确的精确 wheel 安装 exit 0。

同一审查者复核后关闭 I1–I3 与 M1，仅保留合法 `abstract_document` 的 `source_path=""` 被渲染为 `../../../.` 的来源定位缺陷。正常 abstract-only source/synthesis/statistic RED 为 **2 failed in 3.61s**；修复后 focused 为 **2 passed in 3.48s**，完整 `test_writing_workflow.py` 为 **15 passed in 26.18s**，touched Ruff 通过。新的 checkout 外候选 wheel 为 203239 bytes、SHA-256 `3aafafbf7ad1bbe55cd097263b528fae76f741c98fb255536a5e8f2dc861cf97`。定向安装版收据 `<local-workspace>/.runs/academic-agent-v04-final-fix2-installed-20260912-01/installed-abstract-render.json` 为 40 项断言全部通过，SHA-256 `4c1cf3350d71b6095003b7195e87c92b0add8c72e36cf63cbbbb7241fdc5d3d4`：正常 producer 的摘要文档经受控读审、领域和真实写作发布/严格 loader 后，source/synthesis/statistic 都链接同版本 `context.json`，标签明确 abstract-only 和 document ID，目标保存精确 document/pages/evidence 且不是项目根；已有 attached full-text 链接仍解析到实际 immutable blob，正文 unit text 未改。该定向验证为受控 fixture，模型/论文网络调用为 0；未重跑 17 场景或全套。

## 安装后实际产物

安装旅程以原创 RIS 和 A/B 两份原创全文建库，项目一开始明确设置 `max_model_calls=60` 与 `max_input_chars=500000`，没有在恢复时增加预算。实际命令依次覆盖 import、attach、screen、read、review、export、trends、innovate、assess-innovation、domain-extract、writing-outline、writing-draft、人工导入和独立复核。新增语义步骤使用清楚标为 `controlled_exchange_preset` 的 fingerprinted Exchange 响应；严格 Demo 对未知外部包返回失败。它们证明协议、宿主校验、缓存和可见产物，不证明模型语义准确率。

主要正文产物位于原始候选的 `.../projects/writing-original/analyses/writing/v-fb8a028df7a84d87a20a2fef3f124510`。检查了正文中的 A 10→8、B 10→11、bursty 条件、simulated 限制，并从可见 BibTeX 键逐级核对 Work、Document、Claim、Evidence 和独立 review；领域产物 `v-05f5b0409298436191ce9bddc6780160` 及趋势 computation 保留分母 2、算法 `corpus-trends-1`、公式和两项实际 work ID。原来的标题+正文修订会使四个块全部失效，不能证明 mixed provenance；修正轮把它明确保留为整稿失效用例，并新增只改一个正文块的 installed 子版本 `v-72026974f8bf46088d68e849df56356c`。该子版本含 1 个精确作者/时间的 human 块和 3 个保留原作者、模型、时间及经 parent manifest 验证 source pin 的 model 块，独立 reviewer 另行记录。

人工提交文件 及 pending `raw-edit.bin` 的 SHA-256 均为 `5e05b89710264e231fe7941a9f8a7cee5199be7a44bb363e7226211bb98086ff`。捕获发生在任何划分/复核调用之前，保留 BOM、CRLF、emoji 和分解 Unicode；错误数字进入 pending 原文，随后复核标出冲突。W01–W06 分别把错误放入标题、连接语和正文单元，记录完整有序上下文、目标和决定；W07 的原始 bytes 与连续 code-point span 由宿主捕获再交给受控复核。结构拒绝、受控语义决定与实际模型观察在报告中分别标注。

安装旅程实际检查了组合 add/delete/reorder 及 format-only 的稳定身份、显式删除标记和完整当前文稿的有序复核。修正轮另建纯 reorder：pending `v-80f85d4b31ca48c1bf22bb2f6dbf1a9f` 与 reviewed `v-65a9c0f4b86d4a3c8a010bb18c0e3e17` 均为 added/deleted/changed/pending IDs 全空、`reordered=true`、4 个块全部复用；跳过划分，只用一次 fingerprinted 独立整稿复核，block/unit 内容与 IDs、复核新顺序及父版本 bytes 均核对。空正文人工大纲保持 outline，经过 pending→reviewed 后才生成 substantive manuscript。来源、筛选、人工复核和外部 pack bytes 各自使历史产物 stale 且保持完整可读；缺失、格式错误、篡改和越界依赖按相应层拒绝。

外部领域包位于安装核心之外的 `.../external-pack-input/external-domain.json`，SHA-256 `cbf87497a94b86421b1bdaf069d4390131e9948d4a1b7ee47db05bce5bbf59ed`。它保留原 oracle 的 `observation_unit` 与 `transfer_limit`，增加有事实依据的 `arrival_pattern`：A 为 `not_reported`，B 为 `bursty`；比较列顺序和 `boundary_argument` 实际进入领域/写作产物。修正轮在整个 external-pack 旅程前后比较完整的 35 文件安装 payload map，所有代码、领域 JSON、文件集合、bytes 与 hashes 相同；仅明确排除 Python 运行时生成的 `__pycache__`、`.pyc`、`.pyo`。pack 更新后生成 fresh base-only domain，再从同一 parent 显式 `--domain-version` 重绑；正文 raw bytes、作者、原因、old/new pins 保留，非空 trend/candidate/assessment 选择被重新捕获并完成重新划分与全文复核。

重复相同 resume 没有增加成功调用且 ID 相同；预算元数据和日志变化不改变语义缓存身份。语义、pack bytes/path/order、复核和来源变化会失效相应结果。修正轮在每个 `writing-status`、`guide`、项目外 `writing-export-edit` 前后分别保存完整项目目录 topology 与全部 498–500 个文件的 bytes/hash map。主数据库、既有 WAL、配置、日志、元数据、正式产物和其他路径均不变；`writing-status` 的唯一差异是 SQLite `mode=ro` 创建 32768-byte `research.db-shm` lock index 和 0-byte `research.db-wal`。Ruling24 将其记录为引擎 housekeeping，不称作整树逐 bytes 相同；oracle 对非空新 WAL、主数据库、其他文件或 topology 变化仍失败。超过 40k 的 guide JSON 先完整解析，再只截断报告展示文本，实际保留结构化 action/argv。独立固定 60/500000 fixture 在超过 300 字符且含中文、空格、`#`、`%` 的 Exchange 路径完成普通逻辑 CLI 往返；正式产物篡改由 manifest 拒绝。

更详细的安装产物提取在 `<local-workspace>/.runs/academic-agent-v04-release-20260912-01/acceptance-artifact-inspection.json`，其中保留实际目录、结构化文件、断言和可见引用链。

控制器补充的 `<local-workspace>/.runs/academic-agent-v04-fix-round1-20260912-01/root-installed-outline-stale-cache-inspection.json`（SHA-256 `5a0565071fabc2d5142c0c62cb9434387f7ee7203c62af0436e6a4cfce00fb4c`）通过严格 loader 核对安装产物中的 human outline pending→reviewed→nonempty draft；四种独立 screen/manual/source/pack stale 状态均为历史可读、`current=false` 并保存恢复 argv，相同 resume 的有序 unit IDs/review 相等。原 harness 捕获的成功模型阶段/响应数是 22；该补充检查没有新建 DB-before 快照。检查脚本最初把相对 output 解析到错误根目录，随后改为捕获命令的 `--project` 根；这是 inspector 修正，不是产品缺陷。

## 限量真实后端观察

第一次真实运行使用安装环境的 `python -I -m academic_agent.writing_usage`，参数固定为 `--max-new-calls 4`，OUT 事先不存在。输入是安装旅程保存的真实 original-context clone：固定预算 60、已消费或预留 49、两篇原创全文、当前 reviewed outline；使用现有 Codex 登录与经确认的 `http://127.0.0.1:7897` 路由，没有搜索、自动重试或预算增加。

运行在第一个 `domain-extract-v1` 预留和后端启动尝试后停止。报告 `<local-workspace>/.runs/academic-agent-v04-release-20260912-01/.runs/v04-real/report.json` 为 3301 bytes、SHA-256 `a3d60e968f10b9f213be7f962041435c868d2e0b01b86343399c90b58bca0c2f`，状态 `incomplete`，exit 2；reservations=1、attempts=1，预算 49→50，uncertain 1→2。结果错误为 `Codex exited unsuccessfully; check local login, sandbox permissions and CLI compatibility`，domain 没有产物，draft 为 `not_run`。因此不能证明提供方或模型 API 实际收到请求，不能声称完成四次调用，也没有可作人工数值/条件/引用检查的实际模型输出。requested model 为 null，observed model 为 unavailable。

只读根因诊断针对后端实际解析到的原生 `codex.exe` 0.153.4：普通环境的 `codex login status` 返回 `Could not find home directory`；只给该诊断子进程设置 `HOME=<user-home>` 与 `CODEX_HOME=<user-home>/.codex` 后 exit 0 并报告使用 ChatGPT 登录。没有读取/复制认证内容，没有修改全局环境，也没有清理文件或更改 ACL。先前由 Task6 工作进程检查到的 npm wrapper 0.153.2 不是后端实际 executable，不能作为 app 参数兼容性证据。

Ruling22 在用户已知情的情况下允许一次独立、环境已修正的 continuation；原公开参数仍为 4，本次最多再启动 4 个后端，两个 episode 的累积实验上限为 5，且原失败仍保守计数。预检查确认原报告正好 1 reservation/attempt、domain/outline 当前、项目仍为 60/500000、预算 50/60、新 OUT 不存在。第二次运行只在子进程设置上述 `HOME`/`CODEX_HOME`，但仍于第一个 `domain-extract-v1` 后端启动返回同一通用失败并停止：本 episode reservations=1、attempts=1、预算 50→51、uncertain 2→3、exit 2、domain 无产物、draft `not_run`。因此累积为 reservations=2、backend attempts=2、成功阶段 0/4；不能证明 provider/API 是否收到任何请求。组合收据 `<local-workspace>/.runs/academic-agent-v04-release-20260912-01/combined-live-episodes.json` 为 5923 bytes、SHA-256 `2ce5f6b608b9d38f34f3c3ae198bdb02ae8b6371ee7e6942c2912c2f9e8e7e29`。裁定要求 continuation 失败后停止，未发起更多调用。

Ruling23 随后提出一次权限修正的 installed comparison，但平台自动审批在进程启动前拒绝，因为发送原始合成论文/领域配置/大纲/证据到外部 Codex 需要用户对该具体传输的明确同意。当时用户同意仍待答复，因此没有进程 exit、reservation、backend attempt 或 provider call；没有重试、绕过或修改 checkout。第三 episode 保留为 `not_run_platform_rejected`，三段组合收据 `combined-live-episodes-permission.json` 的 SHA-256 为 `ded99c576c6349efe339bf8586e2f7842c9d372d59d06ee308381c4214f411ad`。实验事实仍是两次失败 launch 加一次平台拒绝的 non-launch，E14/E15 不变。

## E01–E15

| 条目 | Task6 结果 | 证据与缺口 |
|---|---|---|
| E01 | 通过（候选安装） | fresh checkout 外 Python 3.14 runtime、锁定哈希依赖、site-packages 0.4.0、pytest absent、17 场景通过；最终发布包/提交仍受后续门槛约束。 |
| E02 | 通过（候选安装） | 34 个模块和领域 JSON bytes 相同，候选 sdist 所需源/测试/锁/固定夹具存在；本文件写入后的最终 sdist 待控制器核验。 |
| E03 | 通过 | 安装 recorder 的完整实际 CLI 链，固定 60/500000，无 Task6 预算增加。 |
| E04 | 通过 | 实际段落、10→8/10→11、条件/模拟边界、引用/证据/review、统计分母/算法及混合来源链均已打开检查；新 installed `report.md` 的每单元宿主脚注显示 citekey、原始 document、claim/evidence 或 frozen computation/trend/source 定位。 |
| E05 | 通过（受控语义） | W01–W07 的原文、位置、完整上下文、逐单元决定和检测层可见；未声称模型准确率。 |
| E06 | 通过 | 提交 bytes/hash 与 pending raw 相同，BOM/CRLF/emoji/分解 Unicode 和错误数字均在模型前捕获。 |
| E07 | 通过 | 组合 add/delete/reorder 与 format-only 保留；独立纯 reorder 在第二次 pending 导入后仍为 pending，add/delete/change/pending IDs 全空、稳定 block/unit IDs、`reordered=true`，随后按新顺序完整复核；多目标 pending 的第二次编辑也保留所有未完成目标。 |
| E08 | 通过 | empty-body outline 的 pending/reviewed/draft 生命周期及 manuscript parent 拒绝可见。 |
| E09 | 通过 | source/screen/manual-review/pack 独立 stale；非法、缺失、篡改、越界依赖失败并保留历史。 |
| E10 | 通过 | fresh base-only domain 显式重绑，保留 raw/作者/原因/old-new pins，重捕获非空 optional pins 后全文复核；候选反例 unread→ready 的完整上下文变化在相同 base-domain 语义 hash 下也清空旧支持并重新全文复核。 |
| E11 | 通过 | 成功缓存复用、语义失效与发布失败/迟到历史可审计；预算/日志不误失效。 |
| E12 | 通过（受控协议） | 外部数据包实际改变字段、missingness、顺序和结构；完整 35-file installed payload map 不变；Demo 拒绝且 Exchange fingerprint 一致。 |
| E13 | 通过（Ruling24 口径） | 每条只读命令均有完整 installed project/topology/DB/WAL/log/artifact 前后 map；应用状态不变，SQLite SHM 与 absent→empty WAL 单列为引擎 housekeeping；结构化 actions/argv、早期参数拒绝及独立60/500000中文/空格/>300 字符 Exchange 路径通过。 |
| E14 | 通过（有界实际观察） | 经独立授权的各轮累计 8 次实际尝试，保留旧失败；所需 domain-extract/domain-review/draft/full-review 四阶段均取得有效产物。最新 P7 正文及全文复核新增 2 次成功调用，保留预算到 60/60，没有自动重试或增加预算。不是单次无中断的全真实链路。 |
| E15 | 通过（有限产物检查） | 实际领域与正文逐项对照合成原文，10→8/10→11、40/60 模拟任务、bursty/missing 到达条件及迁移限制均保留；严格 loader、原文哈希、证据区间和引用解析通过。完整 14 单元复核为 13 supported/1 insufficient，后者在导出中可见，须人工处理；不报告准确率。 |

## 保留的历史证据限制

- Task1 最初两项 RED 的原始时间线证据在中断前丢失；后续 RED/GREEN 和独立复审支持功能，但不能重建并冒充原始 RED。
- Task3 的有效 pre-runtime RED 覆盖候选范围和即时 currentness；两个早期 mutation 用例没有证明 stored-source gap，修正后的 interval-mutation 覆盖在 runtime 修改开始后完成。功能与复审已通过，但时间线限制保留。
- Task4 I1 的第一份夹具先触发 `input_too_large`；`max_input_chars=500000` 和预算 60 是 Task6 新场景一开始的显式配置，不是产品默认值或自动加预算。Task4 的一次诊断 iteration 5 只有控制台输出，没有伪造持久日志。
- Task6 早期曾把夹具 metadata 归一化以匹配 Demo；该尝试不能证明 RIS 导入 provenance，随后已删除并恢复 authentic RIS/import provenance，新增语义阶段改用明确标注的受控 Exchange。
- Task6 `task-6-backends.log` 的实际结果是 36 passed in 5.12s；早期口头的 33/6.21 不准确。iteration 10 实际为 1 failed in 148.40s，iteration 12 为 1 failed in 233.90s；原始缺失的 exit receipt 没有事后重建。
- 第一份 affected final 为 2 failed, 61 passed in 485.78s，原因是 recorder 在 JSON 解析前截断 guide stdout；改成先解析完整 stdout 后，focused 回归通过。`affected-final-2` 在 ruling 前开始并被主动中断，不作为结果。下一次 affected run 为 62 passed/1 aggregate failure：17 个场景各自通过，但重复的 planned 列表顺序错误；改用显式 `WRITING_SCENARIO_NAMES` 后最终 64 passed。
- Task6 fix round1 的第一份集成运行是 1 failed in 280.40s：I1/I2/I4 新断言已通过，但完整树 oracle 把 SQLite `mode=ro` 的 SHM/empty-WAL housekeeping 当成应用写入，且从已用满固定60预算的主项目 clone 不能再完成长路径 Exchange。Ruling24 缩窄 sidecar 例外并批准独立固定60/500000长路径 fixture；没有清理 sidecar、重置 attempts、提高预算或改生产代码。随后 focused 5 passed、源集成 1 passed、fresh installed 17/184/268 全通过。
- Task6 fix round2 保留控制器第一次完整 pytest 的 1 failed/619 passed，不把后续 17 项相关 GREEN 相加或改写成整分支通过。source-only stale 与 source+invalid failed 的生产边界来自已经复审关闭的 Task3 Ruling12；本轮没有为了旧测试修改 runtime。
- 第一次 final Ruff 因从 app 目录仍传入 `academic_agent/...` 路径而得到 6 个 E902；正确 app-relative 范围随后通过。失败收据保留。

发行前剩余门槛是最终 docs-bearing 包核验、Git Flow 推送和正式提交六环境 CI。代码复审已关闭；后文记录最新真实观察的完成及保留限制，早期失败不改写。


## 2026-09-13 真实观察与诊断补充

此前的权限与同意缺口已解决。用户对明确列出的两篇原创合成文献、领域配置、大纲及证据发送到现有 OpenAI Codex 服务作出确认；补充逐字节 FIXTURES 来源核查后，正常平台审批通过。最初的本地读取权限失败没有模型调用；只修正已准备大纲版本的读取权限后保持内容 hash 不变。

P5 安装运行新增 3 次尝试：domain-extract 和 domain-review 成功，draft 原生进程失败，全文复核未运行，exit 2。报告 SHA-256 `b139e15ae2c55c6e25d016af7b62db67dff068e5fa04dcbfbfc85bdfcccdb2b3`。实际领域版本 `v-eed33766c64e4ca587c64601600a098d` 仅包含 A；逐字段、复核理由、原文哈希和证据片段已独立检查。大纲仍使用原先完整两篇的受控领域依赖；没有将单篇实际版本替换进大纲，不能宣称一次无中断的全真实链路。

静态检查发现写作 schema 的 union 形式与提供方文档支持的子集不一致。提交 `db75b343` 将 `oneOf` 改为分支互斥的 `anyOf`，四种标签改为显式字符串单值 enum，宿主模型及证据校验不变。有效 RED 为 4 failed/1 passed，GREEN 为 5 passed，三个相关文件覆盖 66 passed/145.43s，Ruff 通过；首次覆盖命令误用不存在文件名，exit 4、零测试，保留原记录。该变更和有界恢复 helper 均通过独立审查。P6 包的 35 个 source/wheel/sdist/installed payload 完全相同，只有 writing_analysis.py 相比 P5 改动；P6 不是最终发行包。

用户随后明确批准最多新增 2 次的正文/全文复核恢复。P6 实际运行只新增 1 次，Codex 返回成功 JSON，但宿主拒绝，writing_progress 记录 `Writing draft output failed strict schema validation`；按失败即停约定，第二次未启动。报告 SHA-256 `9f19dc9aa5630b515219fffb3653eda6a67434d63b11bc73dbfe2c3c1e9e77fe`，exit 2。累计实际尝试 6 次，其中后端成功返回 3 次、通过所需产物校验的阶段 2/4；项目保留计数 58/60（spent 55、uncertain 3），未重置。计量仍不可得。原始解码回复和字段级错误在 P6 未留存，无法事后重建确切无效字段，也不能将静态 schema 修正宣称为旧 native 失败的唯一原因。完整组合收据 SHA-256 `fa325fdfc1219255429de4f2c830f4128327d5e0326787fc02501ea139ec059e`，位于 checkout 外本地审计目录。

当前离线诊断修正保留严格拒绝行为，只在错误中输出最多 8 个受控位置及 Pydantic 错误类型，屏蔽正文、输入值、异常上下文和未知字段名；模型级校验器可能只能定位到所在单元或块。生成要求补充已有的 kind/reference 一致性、connector purpose 和 heading/paragraph level 规则。四个已知无效回复案例 RED 为 4 failed/4.52s，GREEN 为 4 passed/4.34s；这些案例验证诊断能力，不冒充对丢失真实回复的重现。任何后续真实观察须事先保留解码结果用于离线复查，并继续遵守新输出、显式预算及失败即停约定。

三个相关写作文件的诊断修正覆盖为 **70 passed in 148.22s**、exit 0，Ruff 通过；独立审查无待处理发现。4 项 focused 包含于 70 项中，不相加。诊断和提示修正尚未经过新的真实调用验证，后续需要新一轮有界授权。


## 2026-09-13 最终实际正文观察

用户对新诊断方案明确确认后，P7 安装版完成正文生成和独立全文复核，恰好新增 2 reservations/attempts，exit 0，无搜索或自动重试。报告 `draft-diagnostics-output/report.json` 的 SHA-256 为 `a8d2d69f0cde6931d48df633d1e6fe83812234370b14f8702749a8b33bb90be4`。实际正文版本为 `v-e2ebd0c573944d85ba030cb325cde289`；累计尝试 8 次，后端成功返回 5 次，所需通过阶段 4/4，预算 58→60/60（spent 57、uncertain 3）保留，没有重置。

控制器以安装包只读严格加载实际版本，验证 currentness、两篇原文 SHA-256、每条引用的 Work/Document/Claim/有效 review/Evidence、页内区间及文本 hash、BibTeX 和 Markdown 链接；捕获的请求/解码结果与实际正文及有序 review 一致。全部 14 个单元有独立复核决定。检查收据 `root-actual-manuscript-inspection-20260913.json` 的 SHA-256 为 `7144c7455a4109f4a30b2420efe0d5bb4a080222214b841a63eea8c2cd5cd9c2`，包含原文、正文、逐单元证据和实际导出。

正文准确保留 A 的 40 个模拟任务、10→8/20% 下降；B 的 60 个模拟任务、bursty 到达和 10→11/没有改善。它明确 A 未报告到达模式，B 未给出具体分布，不能据此推断到达模式是结果逆转的原因；两者不能证明现实系统普遍有效或领域整体趋势。13 个单元为 supported。最后一个连接单元引用此前受控大纲的复核记录；独立全文复核输入不含该历史复核收据，因此将它标为 insufficient，导出脚注明确保留理由。控制器没有把它改成 supported；该段应由研究者检查或修订。

此观察仍使用原两篇受控领域/大纲依赖，单篇实际领域版本为独立验证证据；不是所有上游阶段均由真实模型重新生成。英文相邻单元在部分段落中没有句间空格，属于可读性限制，保存的单元文本未被宿主改写。它不影响此次数值/条件/引用检查，也不代表成稿可以不经编辑直接投稿。

当前只证明这组有界合成材料的接口工作和可追溯产物检查，不报告模型准确率、召回率、具体模型身份、token 用量或费用。全部历史 episode 与最新报告的组合收据 SHA-256 为 `8db1558fe1b0f1486ad86865f6ecc1e0d52dd858aadaa1d48c26d3507e5d577d`。本轮后不再需要真实调用。普通沙箱初次读取新产物因 Windows 所有者权限失败，随后同用户只读检查通过；没有改 ACL、源文或正式产物。

独立产物复审核对两篇原文、14 单元、全部来源链和捕获 hash 后，确认 E14/E15 可按上述有限工程口径关闭。规格允许 reviewed/complete 版本包含 insufficient 决定，前提是公开保留；本次没有升级该决定。相邻英文句缺空格列为非阻塞的编辑限制。最终发行构建必须保持 P7 的 35 个程序 payload 完全一致；只因文档冻结不重跑模型、旧安装旅程或源全套。
