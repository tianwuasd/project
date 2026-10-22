# 路线 C：相关容量冲突、联合风险与资源组合价值

日期：2026-10-22。范围：方法搜索、原始材料核验与小实验设计；未修改实验、训练代码或 manuscript，未运行模型或训练。优先结论：**先改价值特征中的风险增量估计并做误差归因，保留现有联合风险接单检查；随后检查多个未来模板是否重复占用同一组资源。** 当前 80.03、79.99、82.17 仅单种子，不能据此判定特征无效、改进显著或具有创新。

## 1. 实际检索记录

使用 `testing_tools/academic_agent` 的 Academic Agent 0.4.0，实际调用 `Runner.search(query, limit=6, cached=True)` 三次，来源为 OpenAlex、Semantic Scholar、Crossref、arXiv；每来源最多 6 条，项目上限 150。底层会自行有限重试；本轮未额外重试限流查询。

| 查询 | 目的 | OpenAlex | Semantic Scholar | Crossref | arXiv | 本次保留 |
|---|---|---:|---:|---:|---:|---:|
| `joint chance constraints correlated risk allocation Bonferroni` | 用方法术语找相关性、风险分配和并集界 | 6，成功 | 0，429，内置重试耗尽 | 6，成功 | 6，成功 | 18 |
| `urban air mobility chance constrained traffic waypoint occupancy time` | 用应用术语找最近邻 | 6，成功 | 3，成功 | 6，成功 | 6，成功 | 20 |
| `joint chance constraints sample average approximation resource capacity` | 用计算瓶颈找SAA/容量约束方法 | 6，成功 | 0，429，内置重试耗尽 | 6，成功 | 6，成功 | 17 |

检索去重候选共 **55**；之后补入 3 条明确标为 `web_supplement` 的题名记录，项目共 58 条。补充不是 Academic Agent 自动发现：Baker/Toomey 由机制定向网页搜索获得，Porras 和 Zhou 来自已有本地线索的官方核验。arXiv宽关键词检索混入金融、生物等无关项，说明本轮召回具有来源噪声；不能将55条称作55篇相关论文。

配置模型调用上限无法写0：本版 `ProjectConfig.max_model_calls` 的 schema 下限为1。因此使用 `backend=exchange, max_model_calls=1`，仅执行搜索、保存元数据、fetch、解析和本地检查；数据库实际 jobs=0、attempts=0，Academic Agent自身未启动模型任务；这不包含本次宿主研究agent的推理用量。原始响应、查询、限流错误和书目快照均保留。

本地项目：`local-only/method_search_2026-10-22/route_c_risk/`。

- `query_1.json`、`query_2.json`、`query_3.json`：逐次摘要和耗时。
- `query_log_full.json`：来源原始返回、完整检索记录。
- `library_snapshot.json`：55条检索书目；`library_snapshot_with_supplements.json`：含补充。
- `download_manifest.json`：3篇全文的 work_id、来源URL、本地路径、页数、SHA256。
- `download_initial_manifest.json`：初次整理使用相对路径错误的记录；实际fetch已成功，之后从Agent内容库正确复制，无网络重试。

网络依次按 decide-proxy-7897 技能分类：四个API域均为 foreign（arxiv命中显式规则，其余公共域默认规则），经 `http://127.0.0.1:7897` 可达；三条arxiv PDF地址亦逐条分类为foreign并经同一代理成功下载。网页工具仅作官方核验/明确标出的补充，不替代本地Agent搜索。

## 2. 必须先明确的现有模型边界

现有约束针对**所有已接任务在任一资源格点发生容量冲突的联合事件**。两个共同情景下的硬容量约束与联合风险程序承担可行性；后者用有限变量消去、向外舍入输出概率上界。模型给定合成的共同潜变量—个体扰动分布，策略不观察潜在标签。

因此，目前待诊断的保守性主要是价值特征中的 `B(A,p)`：用逐格恰满载事件的并集界估算未来模板插入风险，再做 `min(1, 剩余额度/B)`。约束本身已经计算联合事件；“减少Bonferroni保守性”的论文不自动说明现有接单检查需要替换，也不能把改变风险约束的结果当成仅改变特征的结果。

相关性有两层：同一任务标签影响多个格点；多个任务共享潜变量。即使条件于潜变量后任务标签独立，跨时间格点仍可能共用同一个标签。直接乘边缘概率会丢失相关性；任何条件化计算必须对潜变量积分，不能把隐藏标签送入策略。

## 3. 五篇核验材料

### C1. Baker & Toomey (2017)：并集界中的事件重叠

**Efficient relaxations for joint chance constrained AC optimal power flow**。DOI [10.1016/j.epsr.2017.04.001](https://www.sciencedirect.com/science/article/pii/S0378779617301487)；[作者预印本](https://arxiv.org/abs/1608.08247)。work_id `work_7a1f4c39240b41f1`，网页补充。

阅读范围：已保存arXiv v2，8页；读第1–3页，尤其第2页 Proposition 1 和式(2)–(5)，并渲染第2页核对公式；未逐表复核实验。出版年2017，下载稿标注2016，不混写版本。

具体机制：对违反事件 E1,…,Em，用

`P(∪ Ej) ≤ Σ P(Ej) − (m−1) P(∩ Ej)`

代替简单求和。若只知道边缘概率，可用 Fréchet 的交集下界，但稀有事件时这个下界常为0。本文应用是电网过电压的联合机会约束；迁移到这里，应首先用于**特征误差诊断或分组重叠修正**，而非取代既有精确联合检查。

反证/条件：所有格点的共同交集可能为空，而局部两两重叠仍很大，此时全交修正不改善B。不能简单减去全部两两交集并称其为上界（截断容斥的方向不同）。若使用蒙特卡洛估计交集，点估计亦不是认证的交集下界。

最小实验：从已有可行状态抽取200–500个 `(A,p)`，同时计算B、精确新增风险、全交修正和“按同一任务/同一资源段分组”的并集值。输出B与真实增量的比值、排序颠倒率、交集为零比例、每次耗时。不训练。若全交项基本为0，停止沿该公式复杂化。

成本：边缘值已有；全交/分组事件查询需要概率引擎支持。是否便宜取决于因子图诱导宽度，不能宣称多项式普适复杂度。

### C2. Marceau Caron, Savéant & Schoenauer (2013)：航空容量概率的结构化计算

**Computational Methods for Probabilistic Inference of Sector Congestion in Air Traffic Management**。[arXiv:1309.3921](https://arxiv.org/abs/1309.3921)，当前记录无DOI。work_id `work_792979e2b2d349f2`，Academic Agent第二条查询发现。

阅读范围：已保存12页全文；读第2–6页模型与算法。第2页用沿途到达时间的Markov条件分布传播；第3页将飞行存在概率聚合为Poisson-binomial计数；第5–6页说明前向采样、样本缓存和按进入/退出事件排序的扫描算法。没有把期望拥堵成本当成联合违反概率。

可迁移机制：按资源进入/离开时间构造紧凑占用区间；复用相同任务/情景样本，比较插入前后损失；缓存中间概率。对于本模型，先按潜变量条件化再积分是合理方向，但这是本轮推导，论文并不验证当前合成依赖律。

关键差异：文中独立性假设与这里共同潜变量相关的任务标签不同；其期望拥堵成本允许按时间加和，而“任一格点冲突”的概率不能按时间加和。把Poisson-binomial公式直接用于无条件相关标签会产生错误。

最小实验：对既有概率检查器记录因子数、有效占用区间、缓存命中率；只重算候选影响的邻域，用原检查器做对照。先验证风险值和向外舍入范围逐例一致，再比较开销。成本低至中等，主要是缓存键和局部依赖失效处理。

### C3. Porras et al. (2023)：SAA的精确求解，不是原分布的精确替代

**Tight and Compact Sample Average Approximation for Joint Chance-Constrained Problems with Applications to Optimal Power Flow**。DOI [10.1287/ijoc.2022.0302](https://pubsonline.informs.org/doi/10.1287/ijoc.2022.0302)；[arXiv:2205.03370](https://arxiv.org/abs/2205.03370)。work_id `work_eb99a25e260b46bc`，已有本地线索的网页补充。

阅读范围：保存34页arXiv v2；读第1–8页、模型(2)、Algorithm 1 和第2.2节。官方期刊页确认2023年、35(6):1454–1469。未将SAA求解精确性扩展为对真实概率的精确性。

机制：每个情景共享一个二元违反变量，限制允许舍弃的情景数；迭代收紧Big-M，删去可证明冗余的情景约束，再利用k-envelope构造有效不等式。论文明确不研究其SAA解对原联合机会约束的统计逼近质量。

差异与迁移：当前离散合成律已知且存在精确概率程序，并不缺SAA可行性判断。该方法更适合**未来模板组合价值的小规模离线优化基准**：选一批相关情景、所有资源约束共享同一情景变量，测试“每模板单算再加权”是否高估可同时服务的订单。但样本外仍由原联合检查器核验；不能把有限样本满足视为0.01风险保证。

条件：论文有效不等式依赖特定随机行结构，不能把任意无人机占用MILP直接套入k-envelope公式。预算先限制模板数10–30、情景数50–200、固定求解时间；若模型规模本来很小，收紧的预处理可能比直接求解更贵。

### C4. Pang, Low & Duong (2024)：最直接UAM近邻

**Chance-constrained UAM traffic flow optimization with fast disruption recovery under uncertain waypoint occupancy time**。DOI [10.1016/j.trc.2024.104547](https://www.sciencedirect.com/science/article/abs/pii/S0968090X24000688)，TRC 161:104547。work_id `work_c42de5f887634e70`，Academic Agent第二条查询发现。

阅读范围：**官方摘要、Highlights及页面可见贡献段**，没有取得完整PDF，不能标为全文精读。书目没有开放PDF URL，未尝试绕过出版社访问权限。

已核验机制：引入 waypoint occupancy time 表述不确定到达和间隔，借飞行实验数据将概率分离条件改为确定性条件，用层次随机搜索处理扰动后的轨迹恢复。与当前模型的关键差别是轨迹/航点分离与恢复，而非订单逐一揭示下的地面或未来槽接单；本轮不能据摘要断言其是否考虑共享充电资源、组合价值或全局全部事件的单一风险预算。

迁移：把“预约格点占用”与WOT占用随机区间作概念对照；未来有真实时长数据后再研究分布校准。当前合成分布与格点容量结果不构成真实飞行安全证据。最小工作是整理资源、事件、概率口径对照表，成本低；缺全文时不推导其等价约束。

### C5. Zhou et al.：Wasserstein DRO，应列为另一研究支线

**Strengthened and Faster Linear Approximation to Joint Chance Constraints with Wasserstein Ambiguity**。[arXiv:2412.12992](https://arxiv.org/abs/2412.12992)；本地既有书目 DOI [10.1287/ijoc.2024.1073](https://doi.org/10.1287/ijoc.2024.1073)，期刊年2026来自父任务本地记录，本轮仅独立核验arXiv 2024预印本摘要。work_id `work_ed888bc7fc244605`。

阅读范围：**官方arXiv摘要**，本轮未下载其PDF，也未核验期刊最终版本。摘要说明针对右端随机的Wasserstein分布鲁棒联合机会约束，强化特定凸内近似、减少约束并收紧辅助变量范围，且给出其保守性关系。

边界：这解决“未知真实分布/经验样本附近最坏分布”的问题；当前实验指定合成律，替换为DRO会改变研究问题及可行域。只有获得真实样本、明确偏移集合与半径选择，并有样本外校准问题时才值得单开实验。不能引用其“更少保守”来承诺当前价值特征改善。

## 4. 额外检索线索（未作全文证据）

- **Iterative Risk Allocation: A new approach to robust Model Predictive Control with a joint chance constraint** (2008)，DOI `10.1109/cdc.2008.4739221`，work_id `work_67df9fb4a5b74189`。本轮仅书目；可作为风险分配术语入口，不据此复述算法细节。
- **Optimized Bonferroni approximations of distributionally robust joint chance constraints**，DOI `10.1007/s10107-019-01442-8`，work_id `work_fd51cef069194ec6`；Agent元数据年2019，可能为在线发表年，期卷年未独立核验。本轮仅书目。
- **Capacity-Constrained Urban Air Mobility Scheduling** (2022)，DOI `10.2139/ssrn.4175959`，work_id `work_9d3e038d902240a0`。仅书目，题名很接近，但未取得正文，不声称模型与本项目重合。
- **A Real-Time System for Scheduling and Managing UAV Delivery in Urban Areas**，work_id `work_85b6474e0fcf4218`；[arXiv:2412.11590](https://arxiv.org/abs/2412.11590)，首版2024，Agent记录会议发表2025、DOI `10.1109/robio66223.2025.11378031`。仅核验官方arXiv摘要：Airport-Unloading Station系统、连接高层调度与执行、AGV/UAV/人员协作；不把系统实现论文当联合机会约束算法证据。

## 5. 最推荐的两个方案

### 方案一：直接新增冲突概率作特征，冻结所有接单约束

定义 V(A) 为已有任务的“至少一格冲突”事件。若插入动作只增加占用、不重排既有任务，则 V(A)⊆V(A∪p)，新增风险为

`Δρ(A,p) = P(V(A∪p) \ V(A)) = ρ(A∪p) − ρ(A)`。

优先让现有有限变量消去程序**直接计算这个新增事件**，以同一依赖律积分；这可能比两个风险值相减更稳健。若程序只给区间 `[L,U]`，保守增量上界可用 `max(0,U(A∪p)−L(A))`，不能用两个上界 `U(A∪p)−U(A)` 冒充上界。后者只能作为未经认证的排序数值，而且可能受舍入宽度影响。

第一阶段仅做离线状态审计：比较B和Δρ，按剩余风险、候选影响格点数、资源瓶颈、模板重叠程度分层；记录B很大但Δρ很小的状态究竟是否影响动作选择。若B/Δρ接近1或排序几乎相同，则保守性不是当前差距主因，应停止新增复杂概率特征。

第二阶段才比较“旧B特征”“精确增量特征”“旧模型”，保持状态、动作、拒绝检查、训练预算和共同随机数相同，至少多个种子报告配对收益差及区间。若每次特征要做K个新查询，应测真实延迟，不预设能在线承受。`min(1,剩余额度/Δρ)` 仍是启发式：多次插入风险一般不线性，不能据此解释为可安全接入任务数。

### 方案二：把未来模板当一个共享资源组合，量化重复计算

当前“各模板价值×权重再求和”可能对同一出发槽、无人机返回窗口和充电占用重复估价。先建立小型未来模板集合，比较：逐模板独立可插入价值总和；资源共享的模板组合LP/MILP；实际顺序插入且通过原风险程序的价值。每个未来订单需至多选一个模板，所有模板共享资源约束；这一步比只改并集界更可能解释多资源互补性，但仍需实验。

如果采用情景模型，必须对同一情景共用违反变量，保留跨资源联合事件；参考Porras的是建模与求解结构，不是直接套其有效不等式。可以先用已存在条件LP82.17作为待解释对照，审计它与当前加权模板的资源口径是否一致，避免重复开发。选择方案后做2×2消融：B/Δρ × 独立模板求和/共享资源组合。若仅组合改进有效，则主因是资源价值重复计而非概率并集界。

这两个方案都属于本轮**待验证的迁移设计**，不是论文已经证明对本项目有效，也不证明新颖性。DRO、SAA替换约束、真实飞行校准另列支线；本轮不与仅改价值特征混合。

