# CT完整母版逐项验收：v0.3.0

核对日期：2026-10-26。母版为server-engineering随包CT 0.22.0，提交 `46b5425ffc634b637fc1715e51627a58a48ea36f`。本表取代v0.2设计中排除c/w/v的决定。采用表示有实现与本地证据，不表示服务器已部署或真实GPU实验已通过。

| 母版部分 | Vista实际入口/方法 | 检查证据与边界 |
|---|---|---|
| 中文面板/导航 | task_panel：s/g/m/l/p/o/h/n/r/k/x/u/c/w/v/e；b返回、q退出、0保留字段含义 | test_task_panel、test_panel_progress |
| 新建顺序 | 名称→步骤→路径→历史输入→模型/范围→GPU→冻结预览 | test_strict_panel；取消不保存半份设置 |
| 路径 | server_profile/servers/zmic44.json展示数据、环境、缓存、模型、预测、报告 | CLI > 原runtime设置 > profile；历史确认与新目录约定分开 |
| GPU表单 | 展示编号/型号/显存/利用率/进程/可见范围→并发上限1–5→候选auto/编号/UUID | test_strict_panel；上限与候选池数量独立 |
| GPU配额 | CUDA/NVIDIA/SLURM限制取交集，冻结UUID，真正执行前重查 | GPUSelectionTests；不清空继承限制 |
| 弹性调度 | train/predict有一张就运行，无卡等待，后来空闲卡加入剩余工作 | test_evaluation_parallel、test_inner_contracts；无合并显存 |
| CPU前置 | 环境和预处理不等待GPU；查询不安装依赖 | test_task_worker，`python -S`只读入口 |
| 独立任务 | 唯一目录、plan摘要、配置、code_manifest及源码快照 | test_task_panel、test_task_lifecycle |
| 后台与锁 | runtime排他，子进程继承锁并登记boot/start/PID | Windows模拟通过；Linux flock/孤儿回归待Linux执行 |
| 中断/退出 | 显示目标再STOP，仅身份匹配进程，保留完整检查点 | test_task_posix、test_panel_linux（Linux专用）；退出面板不停止后台 |
| 运行中更新 | u确认停止目标并等待实际退出，然后备份更新；不自动重启实验 | 生命周期与锁路径；不隐式强杀/解锁 |
| 恢复分级 | simple只查未完成/追加步骤直接前置，小文件SHA、大文件size+mtime；full查全部；none跳产物 | test_inner_contracts；范围/缺失凭证/字节/速率/ETA写resume-check.json；源码/plan不跳过 |
| 来源证据 | plan/state/code_manifest/receipts摘要选择后冻结，复制前再查，input_evidence保留原件 | test_inner_contracts；旧工作区无这些证据时不称完整核验 |
| 输入复用 | 独立复制，不用硬链接；兼容性失效有界新建；权限/数据/磁盘错不吞掉 | test_task_lifecycle；来源字节不变 |
| 重跑/追加 | 重跑新目录，仅重算prepare才新缓存；上游重算后要求相应预测/汇总；追加保留成功步骤 | test_append_authorization、test_task_lifecycle；缺原GPU分配不能直接追加GPU步骤 |
| 环境 | 原runtime、独立Conda prefix与缓存；可复用既有Conda程序；固定版本/pip check | test_environment_progress、test_panel_linux；未在zmic44执行新安装 |
| 安装进程 | 下载字节→SHA→Python→PyTorch→nnU-Net→版本/依赖核验 | environment.json、launcher.log；失败不写成功凭证，不伪造解压百分比 |
| 软件检查 | software_smoke用生成的12³ CPU张量训练/写读检查点 | 本机实际通过；software_only不签发正式训练资格 |
| 正式尺寸预检 | train/resume绑定实际GPU UUID/硬件、fold/模型、plan、环境、源码和CPU预算 | preflight_gate门禁复用/失败/篡改测试；不能从train入口绕过 |
| 预检方法 | 实际patch/batch，3步train、2步val、权重/优化器加载及恢复后一步 | smoke_train；真实GPU执行待服务器，不是精度验证 |
| 有界CPU选参 | 实际GPU上比较1/2/预算内最多3个线程候选，2步热身后计时；保存报告/SHA/有效配置 | test_inner_contracts；短样本不称全局最优；续训固定已选参数，配额不足明确拒绝 |
| 线程落实 | threadpoolctl、Torch、NumExpr及四个env同步，正式训练持有limiter | 合成回归及实际CPU线程池验证 |
| 增强进程差异 | nnUNet DA workers保留0 | 保留现有B0串行采样/全局RNG恢复契约；不搜索多增强进程，不称与CT加载进程调优完全等价 |
| 资源探测 | affinity/cgroup CPU与内存、0.2s空闲采样、load、RAM、共享内存、I/O PSI | task_resources；未知值保守回退，非整机全开 |
| 病例预算 | 原图/重采样header体积×128B+1GiB估计，最多75%可用RAM；高I/O压力最多2病例 | cpu-case-budget.json；有界估算，非实测峰值/吞吐最优 |
| CPU病例并行 | cpu_case_queue预处理独立病例，最多8进程、每进程1线程；继承锁/登记PID/失败清理 | 资源/调度回归；真实nnU-Net并行待服务器 |
| 科学方法 | 仅8fit计算指纹、4cal不进入指纹；官方归一化、七类映射/空间单位不变 | cohort/baseline契约；不套用CHD不同解剖算法 |
| 训练 | 每折一个B0 nnU-Net、跨折共享上限；CT250epochs、来源开发EMA Dice选best | server_train/project_trainers；没有CHD五阶段或LSTM依赖；未训练真实病例 |
| 检查点 | 原子替换，latest/final继续，不从best/短测续训；保存Python/NumPy/Torch/CUDA全局RNG | CPU RNG往返验证；不承诺数值逐位一致；旧权重无RNG时明确记录 |
| 预测粒度 | 每卡一个病例，使用冻结fold/B0/best；单折也可多病例并行 | test_evaluation_parallel；正确模型映射不变 |
| GPU→CPU尾部 | 官方预处理→logits落盘；全部GPU进程退出后再恢复原网格/压缩/LCC/Dice/HD95 | test_inner_contracts验证阶段顺序、CPU隐藏GPU；真实API集成待服务器 |
| 中间结果恢复 | logits/properties分别SHA，绑定数据/模型/病例/冻结清单，manifest最后提交 | 篡改测试；CPU尾部可复用GPU产物而不重新申请卡 |
| 旧已完成病例 | 复用前核对签名和输出摘要 | evaluation_contract；不靠running/临时文件判断完成；raw主结果/LCC次要分析不变 |
| 完整进度 | s/t显示步骤、安装、准备病例、预算与预检；m固定模型指标表；l分层日志 | test_panel_progress；一次查看/持续刷新，发现新日志 |
| 排队模型 | model_selection+gpu_queue合并；epoch/train loss/val loss/lr/UUID；未开始折也显示 | test_inner_contracts；未记录值为— |
| ETA | 恢复仅用本attempt计时；预测仅用本轮phase样本；等卡/失败不估确定ETA | test_inner_contracts；旧case时长不冒充本轮吞吐 |
| 结果中心 | 简洁摘要、病例/七结构、训练/来源/文件；详细与分页 | test_task_panel；只读旧结果，不改变last-job |
| c网络 | 原runtime非秘密设置；官方HTTPS/SSH；zmic建议SSH443+17891+已有部署密钥 | CT network.py适配；取消不保存，不改系统代理 |
| u更新 | 官方及明确历史镜像身份；下载仅官方，校验URL重写；备份/受管release | test_private_update/test_task_lifecycle；旧入口可直接Git引导升级 |
| w代码上传 | 逐提交核范围/对象体积，预览已提交Vista代码，main快进、核远端、写回执 | test_publication本地Git；不自动commit/强推，不带其他工程或研究归档 |
| v产物上传 | 完成且稳定的任务；预览配置/日志/指标/完成权重；同SSH+LFS独立分支 | test_publication；排除04_data、08图像、失败恢复归档、未完成模型 |
| 上传重试 | u/w/v同runtime网络；不可变分块包核验、同包重试与回执 | 篡改/重试测试；实际私有LFS写权限待服务器核实 |
| 工程记录 | 本表、SERVER_PANEL、SOURCE.json、HANDOFF/STATE/DECISIONS | 不改其他工程或skill母版，未产生跨工程实测优化结论 |

## 来源与业务适配

发布核心逐文件来源及适配摘要见 `05_src/panel_publication/SOURCE.json`。预测拆分核对了官方nnU-Net 2.8.1 [predict_single_npy_array](https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/nnunetv2/inference/predict_from_raw_data.py) 与 [原网格恢复函数](https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/nnunetv2/inference/export_prediction.py)，只移动官方调用的执行阶段。

CHD专属疾病诊断/LSTM/五阶段模型不适用当前B0。GPU随实际计算进程退出释放，没有新增跨步骤显存holder或承诺独占预约。旧任务快照不静默迁移；旧模型不同代码签名仍拒绝混用。

## 验证状态

机器可读结果见 `STRICT_VALIDATION.json`。独立只读审查已复核线程池、排队模型展示、I/O约束三项修复，未发现已确认的P0/P1阻断错误。

当前Windows环境跳过Linux专用进程测试和需要影像依赖的测试。真实nnU-Net API集成、真实GPU尺寸/峰值、服务器安装、实际代码/产物上传仍需服务器验收。本次[GitHub Actions](https://github.com/tianwuasd/project/actions/runs/40211433316)仍因账户计费/支出上限在启动前失败，执行步骤为0；平台未启动不算测试通过。代码已发布并核验main提交 `de307c6e98355c5e077a65947ed29412662df221`。
