# V2 模板映射及适配边界

工程模板：server-engineering / VISTA 0.5.1，提交
`4221c35e7a55a3a3a9c456b9fd66c6b18fc3bcc5`。
科学代码：CT 0.24.0，提交 `b7c6881bd7fdff7d5935d772c2d6c001e653bf9f`。
`panel_v2/TEMPLATE_SOURCE.json`记录直接复用的资源探测模块 SHA256。
`CT-MASTER-MANIFEST.json`是原始 CT 模板的来源证据，不是修改后 V2 的校验表。

| 模板能力 | CT V2 实现 | 核对要点 |
|---|---|---|
| 独立阶段/来源图 | panel_v2/panel.py、service.py | prepare→train→test 或 predict-all→diagnosis；显式来源，禁止隐式最近任务 |
| 只登记本代任务 | identity.py、catalog.py | 根标记、同实例、路径不重叠、不接管非空旧目录，拒绝链接 |
| 冻结源码/输入 | snapshot.py、server/jobs.py | 分离运行源码与可更新代码；外部配置复制，原文件变化不改已提交任务 |
| 独立环境/修复 | panel_v2/environment.py、server/bootstrap.py | 每 release 单独环境；每次实际导入/API调用，直连后本V2代理一次重试，依赖版本漂移拒绝 |
| CPU80/RAM80 | live_resources.py、resources.py、cpu_runtime.py | 当前 affinity/祖先 cgroup/空闲容量；已知零阻塞，未知保守回退；不跑候选排名 |
| 训练实际加载 | imagechd/train.py、loading.py | 预算进入实际持久 train/val 池；RAM两池相加；预取1，整例解码计入估算；训练配置与资源选择分离 |
| 读取与计算分开 | read_io.py、common.py、preprocess.py | payload读取/解压持本地锁，变换在槽外；阶段入口 PSI 高时限制读槽，保留计算池 |
| GPU 获准范围/外部进程 | server/gpu_hold.py、resources.py、imagechd/*queue*.py | 继承可见范围、排除外部进程、独立网络/折队列；当前模型批量不变 |
| 显存保护 | imagechd/gpu_memory.py | 0.24预算、保留余量、阶段/迭代检查、OOM失败记录；不保证其他用户突然抢占时绝不OOM |
| 恢复/停止 | catalog.py、service.py、server/jobs.py、resume.py | 全量来源/快照校验；身份核验后停止；失败保留检查点；已完成任务 k 新建 |
| 回收/恢复 | catalog.py | 活动和有下游的任务拒绝；先持锁登记再改名；恢复原路径；不物理删除 |
| 查询/结果 | panel.py、server/inspection.py、result_views.py | 只读，分区及简洁/详细；完整每折结构与病例结果由CT原展示器解释 |
| 日志/诊断 | panel.py、service.diagnostics | launcher/阶段/模型日志，资源JSON；本地ZIP、相对路径、不混同名日志、凭据脱敏、变化/超限拒绝 |
| 更新/上传 | server/updater.py、code_sync.py、ssh_upload.py | u/w限定codex/ct-panel-v2且只允许V2路径；v独立chd-v2-artifacts分支；SSH+LFS，无gh |
| 独立小规模测评 | service.STAGES.assessment | 独立缓存/任务，软件smoke+预测评估；不向正式任务复制workers/权重/测评结果 |

## 与 VISTA 示例不同的部分

- CT 的六网络/三折独立实验队列保留；没有把它称为 DDP，也没有引入 nnU-Net、AMP、自动 batch、compile 或250轮参数。
- 资源在阶段入口及恢复时重新计算，数据加载器在本网络的整轮恢复边界重建。未迁移 VISTA 的训练中滚动扩容和动态 DDP 成员变更。
- CT逐病例缓存与模型恢复沿用0.24事务；未声称实现VISTA独有Blosc惰性patch、CUDA双流pinned传输或持续NUMA亲和优化。
- 内存是整例/批量的保守估算；未迁移VISTA按病例RSS高水位反馈。预算并非操作系统预留，仍须目标服务器实测。
- a是CT的软件/资源小规模测评，不是模型质量证明；真实规模预检仍在正式train之前按CT原流程执行。正式任务不复用a的产物。
- 重跑与旧任务恢复不同：k以当前代码创建新任务；r核验原代码与依赖继续原未完成任务。更新u不改冻结任务。

本轮不修改已安装 skill 的默认模板，避免将未合并的 CT V2 实验面板变成其他工程默认值。
