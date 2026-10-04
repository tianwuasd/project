# 最邻近工作全文核对

2026-10-04。物理页码；仅把实际读取范围用作证据。下载原文保存在本地 `fulltexts/`，不进入公开发布包。

| 工作/证据 | 已确认内容 | 对本研究的约束 | 本轮仍可检验的差别 |
|---|---|---|---|
| [Chen, Ulmer & Thomas, EJOR 2022](https://arxiv.org/abs/1910.11901)，预印本物理5–11页，8页已看图 | 随机请求；车辆/无人机接单；硬期限；接受但未装载客户保留；动作更新计划；无人机单订单往返且返回充电；DQL用资源未来可用时间 | “硬承诺+灵活计划”、后决策状态、DQL派单不是新概念 | 本轮显式联立有限起降、机队、充电的情景资源日历；所读模型中未见公共扰动的联合返航预约。仍须核对其他文献，不能据此称首次。 |
| [Chen, DASC 2019](https://junchen.sdsu.edu/proceedings/dasc19_chen.pdf)，物理2–4页，4页已看图 | 路由、时间窗、SoC、充电功率与充电位联合约束；式16限制同时充电；MILP变换 | 充电位、动态充电或路径耦合不能单独列为创新 | 本轮是未知订单顺序下的不可撤销接单与联合情景预约；非该文完整异质机队路由模型的替代。 |
| [Asadi & Nurre Pinkley](https://arxiv.org/abs/2105.07026)，物理6–7、10–11页文本 | 随机换电需求，充/放电和换新；电池数量/容量上的单调值函数；单调ADP及回归初始化 | 单调值函数或单调ADP已有充分先例；不借用该文的假设或定理 | 本轮检验跨时段、跨资源任务模式的互补，而非电池库存单调性。未复核该文完整证明，不能声称已重现。 |
| [Ren et al., Energy-Predictive Planning](https://arxiv.org/abs/2508.01671)，物理9–14、26–27页文本 | 天空航路、有限补能节点、能源预测及到达/充电预约；预测误差可导致悬停/延误 | 能源预测、有限补能与预约组合本身不是新贡献 | 本轮每个预定联合情景都要满足返航落地资源约束；仅对所列情景有保证。该文实验含实验室小型无人机及缩放，不作为室外物流参数真值。 |
| [NASA, Vertiport Dynamic Density, 2023](https://ntrs.nasa.gov/api/citations/20230012622/downloads/Vertiport_Dynamic_Density-TM-20230012622.pdf)，物理5、7–8页 | 不确定到达/容量和有限电量使拥堵缓冲与等待/备降决策重要；过量缓冲可能产生多余延误 | 支持现实动机，不提供本研究共同扰动分布或安全认证 | 检验不同预约表达下的容量损失。不能据此声称无需缓冲即可安全运营。 |

## 尚未取得全文的关键邻近研究

- [Hu, Du & Li, TRC 2025, 105381](https://doi.org/10.1016/j.trc.2025.105381)：当前仅摘要，已覆盖风险感知、多阶段多智能体、骑手无人机协同。多站MARL不能直接当空白。
- [Campuzano et al., ICCL 2022](https://research.utwente.nl/en/publications/the-dynamic-drone-scheduling-delivery-problem/)：摘要含中央站、随机包裹、时间窗、能耗及值函数RL。机构公开PDF返回403，未当作全文已读。
- [Robust optimization for truck-and-drone collaboration with travel time uncertainties, TRB 2026](https://www.sciencedirect.com/science/article/pii/S0191261525002279)：检索发现，出版商页面读取失败；相关不确定性、鲁棒路线的精确差别待核查。
- [Integrating urban air mobility into the power grid through smart charging solutions, TRC 2025](https://doi.org/10.1016/j.trc.2025.105281)：当前出版商摘录，不能以灵活充电窗口/随机充电调度本身宣称创新。

## 方法归属与限制

[OptNet（Amos & Kolter, 2017）](https://proceedings.mlr.press/v70/amos17a.html)已经把可微优化作为网络层；[Zanon、Gros等的MPC研究](https://cse.lab.imtlucca.it/~bemporad/publications/papers/ecc19-Qlearning.pdf)已结合Q学习和预测控制。本轮资源模式特征并非可微优化层，不能把“优化+学习”包装成新方法。尚须系统比较组合资源特征、网络收益管理和近似动态规划的相关工作。

结论：可继续开发候选机制并证伪，但全文覆盖尚不足以确认建模或方法创新。当前最可靠的新证据是本研究首版分离值函数的表达局限及待验证修正，不是文献首次性。
