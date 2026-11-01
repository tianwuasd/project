# CHD V2 2.2.0：main中的独立V2入口

2026-11-01按用户要求合入main。main同时包含V1的chd-ct-reproduction和V2的chd-ct-panel-v2；V1现有文件、原启动命令和科学方法保持不变。

## 从原服务器V1更新后打开V2

先在原V1面板使用u → 1更新main，退出面板，再进入它旁边的V2目录：

```bash
cd ..  # 从当前chd-ct-reproduction进入仓库根目录
# 原先使用稀疏检出的仓库需要补上新V2目录；普通检出无需处理
if [ "$(git config --bool core.sparseCheckout)" = true ]; then
    git sparse-checkout add chd-ct-panel-v2
fi
cd chd-ct-panel-v2
bash start_server_v2.sh
```

应看到 `CHD V2 2.2.0 | 选择分割方法`：选1进入STU-Net-S，选2进入原CT六路。V1目录内没有此新脚本；只更新main后继续打开原V1入口，仍会进入V1。

| 面板 | 同一main检出下的目录 | 启动 |
|---|---|---|
| V1 | chd-ct-reproduction | 原 `bash start_server.sh` |
| V2 2.2.0 | chd-ct-panel-v2 | 新 `bash start_server_v2.sh` |

```bash
bash start_server_v2.sh --engine stunet
bash start_server_v2.sh --engine multistage
bash start_server_v2.sh --help
bash start_server_v2.sh --engine stunet --query history
```

Windows可用start_server_v2.cmd或python start_server_v2.py。原V2/start_server.sh仍是六路兼容入口。打开和查询不安装环境或启动训练；正式计算目标Linux x86_64。

u/w支持main以及原V2开发分支，始终跟随实际检出的分支，不替用户切换。更新仍保留“拒绝V2之外路径变更”的保护：若同一main还包含其他项目的待更新改动，V2会停止，需在原仓库人工核对更新。V1和V2共享同一代码检出时不要同时执行代码更新；两面板的任务、环境和结果仍独立。

本次main交付核查见[2.2.0说明](docs/V2_2.2.0_MAIN.md)。STU-Net原论文及代码一致性边界见[科学复核](docs/STUNET_SCIENTIFIC_AUDIT.md)：官方架构/权重一致，新训练按作者公开代码1000轮，保持用户batch4和nnU-Net2.8.1兼容适配，尚非严格原论文复现。

## V2内的方法与数据

- **STU-Net-S（新方法）**：S1官方预训练微调，S0同架构随机对照；七结构标签顺序保持ImageCHD定义。额外标注编号为ignore=8，模型仍输出背景+七结构共8类。采用作者Excel间距派生副本、单例Z-score、训练子集拟合规划、3D patch及全体积滑窗；不使用原六路硬投票。
- **原六路方法**：保留CT0.24的六网络、三折、预测ROI及验证后处理合同。没有向它套用STU-Net的优化器、1000轮、AMP/DDP等参数；V1的0.25等更新也没有整仓合并到本分支。
- 两种方法只读源影像可共用；任务索引、缓存、权重、环境、日志和结果按方法隔离。V1或旧独立STU-Net运行目录不能直接接管。原CT V2已登记任务仍按原冻结代码恢复。

## STU-Net-S的使用顺序

选1进入新方法面板后：e准备环境 → 1预处理 → 2训练（默认S1）→ 3外折预测与评价。默认三折；用户选全局batch，默认4，扩卡不增加批量。真实训练前会检查所选模型的实际尺寸和容量。

输入为ImageCHD私有数据描述目录，不是V1缓存。生成方法与权重出处见 [STU-Net输入说明](foundation_engine/README.md)。旧独立工程的**数据描述**可作为只读来源；其缓存、任务或模型不迁入V2。缺少官方权重或SHA/骨干层不符会停止，不静默随机训练。

```bash
export IMAGECHD_STUNET_WEIGHTS=/your/local/weights/small_ep4k.model
bash start_server_v2.sh --engine stunet --stage prepare --data-dir /your/private_input_descriptor --models S1 --non-interactive
bash start_server_v2.sh --engine stunet --stage train --input-job /completed_V2_STU_prepare_job --models S1 --global-batch 4 --non-interactive
bash start_server_v2.sh --engine stunet --stage predict --input-job /completed_V2_STU_train_job --models S1 --non-interactive
```

## 路径隔离

| 方法 | 环境/任务登记 | 任务结果 |
|---|---|---|
| V1 | 沿用原V1配置 | 沿用原V1配置 |
| V2原六路 | chd_ct_v2_runtime | chd_ct_v2_results，预处理另在chd_ct_v2_prepared |
| V2 STU-Net | chd_ct_v2_stunet_runtime | chd_ct_v2_stunet_results，各任务workspace内保存预处理 |

Linux默认路径见各方法server配置；自定义目录必须显式一致，首次使用空目录。整合版STU-Net会写专属身份标记，拒绝已有外来状态、目录重叠、符号链接/联接、不同实例混用。未初始化的查询不写任何目录。两个面板可同时打开；计算仍按获准GPU和外部进程占用准入，不抢占另一面板或其他用户的任务。

## 更新、来源与验证

u/w跟随当前检出的main或原codex/ct-panel-v2分支，仍限定整个chd-ct-panel-v2代码子树；不会切分支，不走独立imagechd-stunet更新线路。代码更新不热修冻结任务。v仅按明确选择上传产物到独立分支，代码上传会拒绝嵌套私有清单、影像、权重和运行输出。

科学方法与工程模板分开记录：STU-Net资源引擎来自server-engineering Skill v2 / VISTA0.7.0固定提交21ad9e539cb7dbcc6937b7e09064a7261cd2b913；STU-Net源来自imagechd-stunet提交01bf801cd4dc9d8da885283946b8befeaf8d9ec2；原六路算法仍CT0.24。不是把STU-Net称成VISTA foundation。

本轮Skill符合性、冲突修复、检查结果和未验收范围见 [2.1整合核查](docs/V2_2.1_INTEGRATION.md)。没有启动正式三折训练，没有新的外折准确率；本机合成/Gloo验证不能代替服务器Linux/CUDA/NCCL验收。

---

## 原V2六路方法的2.0.x说明（历史与兼容参考）

面板版本 2.0.2，科学流程基于 CT 0.24.0。保留六网络、三折、预测裁剪框、加权投票、验证集决定的后处理以及显存保护。工程参考 server-engineering / VISTA 0.5.1；具体映射和未采用的算法特性见 [模板映射](docs/V2_TEMPLATE_MAP.md)。

2.0.1修复显存保护接收无编号 `cuda` 时的启动错误，按worker当前可见设备解析编号；不改变科学参数和缓存格式。更新后新任务使用修复代码；旧任务的 `r` 仍使用原冻结代码，遇到此错误应更新后用 `k` 独立重跑，保留旧任务证据。

本目录只在 `codex/ct-panel-v2` 分支开发。V1 的 `chd-ct-reproduction` 保持原样，main 不合并本分支时，原服务器的 main 更新不会取得 V2。

2.0.2：显存使用80%软预算，超过后可继续使用剩余空间；取消人为分配上限，保留外来进程检查和实际空闲低于0.5 GiB时的停止检查。模型、batch和训练参数不变；不新增自动调batch。V2旧冻结任务须用k独立重跑，新代码不会修改旧快照。

## 开启

在单独检出的 **codex/ct-panel-v2** 工作目录中进入本目录：

```bash
bash start_server.sh
```

不要把服务器现有 V1 checkout 切换到 V2 分支。需要尝试 V2 时，应另建 checkout；原 V1 启动命令保持不变。查询与菜单只依赖 Python 标准库，实际计算环境在首次任务时安装到独立 runtime。安装目标 Linux x86_64 / Python 3.11 / Torch 2.8.0 cu128；其他架构须另行适配。

默认服务器目录：

| 用途 | 路径 |
|---|---|
| V2 环境、缓存、登记与冻结代码 | `/data5/zhougaowei/zhangruichen_workspace/chd_ct_v2_runtime` |
| V2 任务与结果 | `/data_nas/zhangruichen/chd_ct_v2_results` |
| V2 预处理缓存 | `/data_nas/zhangruichen/chd_ct_v2_prepared` |

首次使用必须是新目录或已登记的同一 V2 实例；拒绝非空旧目录、交叉根目录和链接。环境按冻结代码 SHA 分开保存在 `runtime/envs/chd_v2_py311/<SHA>/`。原始影像可只读共用，V1 的任务、预处理、模型和网络偏好不导入 V2。

## 菜单和顺序

| 入口 | 用途/输入 |
|---|---|
| e | 检查/安装 V2 环境，实际导入依赖并调用资源 API |
| 1 | 新建预处理，选择原始数据；生成 V2 自己的缓存 |
| 2 | 选择已完成的 V2 预处理任务，创建正式训练 |
| 3 | 选择 V2 训练任务，执行 CT 原测试预测/评估协议 |
| 4 | 选择 V2 训练任务，预测全部病例，供诊断特征使用 |
| 5 | 选择第4类任务和诊断表，导入标签→特征→诊断树→解释→评估 |
| 6 | 选择已完成的第3/4类任务，单独重算评估，不重新预测 |
| a | 独立小规模软件测评：最多6例的准备、smoke训练、测试；产物不能作为正式任务来源 |
| s/g/m/l/p/h | 进度/显卡1秒刷新/模型记录/跟随日志/本人进程/登记历史 |
| o | 分割、训练、决策、诊断评估、文件五个分区；简洁/详细 |
| r/k | 恢复未完成任务 / 按当前代码独立重跑，保留旧任务 |
| t/d | 停止本任务 / 回收或恢复；活动任务及有下游依赖的任务不能回收 |
| z | 导出本地诊断 ZIP，不上传；超过32MiB或日志变化则拒绝并说明 |
| c/u/w/v | 本V2网络配置 / 更新V2分支 / 上传已提交V2代码 / SSH+LFS上传产物到独立分支 |

普通输入页 b 返回，q 退出；退出面板不停止后台任务。Ctrl+C退出日志/显卡查看。各阶段提交前显示来源和目录并确认。状态 passed 只表示流程完成，不能等同于分割质量达标。

## 隔离、资源与恢复

- 只发现 `runtime/pipeline_v2/index.json` 中登记的任务。来源必须显式指定且成功完成；摘要变化、缺文件或活动进程均拒绝复用。
- 每个任务冻结完整运行代码、配置与来源。后台用快照启动；恢复沿用同一快照，环境依赖版本变化会报错。新的代码更新只作用于后续新任务。
- 当前可用且获准 CPU 的80%向下取整、可用RAM80%决定实际病例/加载进程；已知不足就阻塞。未知资源才保守回退并记录。
- 训练两个持久加载池同时计入RAM，读取槽与计算池分开；每个并行CT实验分摊资源。资源选择不改变CT原批量、轮数、优化器、标签、划分或损失；原seed配置保留。
- 沿用 CT 的 GPU 候选范围、外部PID排除、跨折/网络队列、显存安全余量和GPU结束后CPU收尾；不能把它称为DDP或保证绝不OOM。
- 环境安装直连优先，失败后只使用本V2的 c 代理重试一次。网络配置不改变全局代理，也不修改 V1 设置。SSH密钥仅记录路径，不读取或复制私钥内容。

## 脚本与复现

```bash
python3 start_server.py --help
python3 start_server.py --query history
python3 start_server.py --stage prepare --dataset /your/raw/data --name V2准备 --yes
python3 start_server.py --stage train --input-job <V2预处理任务ID> --name V2训练 --yes
python3 start_server.py --query results --job <V2任务ID> --result-section segmentation --result-detail detailed
```

自定义实例必须同时指定 `--runtime`、`--results`、`--prepared-root`。任务ID来自h历史，不是V1目录。`--preview`保存冻结计划并打印，尚不提交计算。`r`不追加其他阶段，下一阶段应另建任务。

仅验证软件的合成测试使用 `--stage synthetic --device cpu`；不要把合成结果当真实CT效果。正式数据不随代码分发。科学流程/公式/限制可查 [CT0.24参考](docs/CT024_REFERENCE.md) 与 [ROI/显存说明](docs/roi-refinement-guard.md)；继承文档中的旧面板命令不是V2入口，应以上述菜单和 `--help` 为准。

开发验证：在专门环境中安装本项目与 `.[dev]`，禁用CUDA后运行 `pytest`。来源清单/实际验证记录见 `V2_DESIGN.md` 和 `docs/V2_VALIDATION.md`。Windows合成、文件校验及本地Git验证不能代替 zmic44 的 Linux 进程锁、实际CUDA显存与真实数据验收。
