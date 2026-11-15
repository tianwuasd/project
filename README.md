# 先心代码助手3：小样本完整流程

独立工具目录、独立Git仓库、独立分支 `codex/chd-segmentation-mini-workflow`。通过固定版本的外部生产项目接口，运行2例训练、1例验证、1例内部测试，收集完整流程的真实耗时、恢复结果和错误记录。

| 项目 | 源码及更新范围 | 运行配置与日志 |
|---|---|---|
| 助手1 | `chd-segmentation`，STU-Net及共享生产接口 | 由助手1自身配置指定 |
| 助手2 | `chd-segmentation-assistant2`，MedSAM2与Q | 由助手2自身配置指定 |
| 助手3 | `chd-segmentation-assistant3`，本工具 | 私有`project.local.json`及本目录`.runs` |

助手3的更新器只接受本工具自己的独立Git根目录。运行时核验生产项目完整提交及实际源码字节，并在各阶段前后和最终收尾核验原始清单、冻结划分及权重摘要；生产项目更新后，需要重新核对接口兼容性，再明确修改依赖提交。本工具不切换、合并或更新助手1/2的工作目录。

## 本机使用

本项目沿用生产项目已有Python环境，工具自身使用psutil、filelock、joblib及fsspec管理进程和运行记录。`project.local.json`包含机器路径及输入来源，加入忽略清单。公开模板见`configs/example.json`，配置schema为2且data_mode为offline。

生产接口固定于公开提交 `277c11a97ab0f2d666d1d8e669fcdb6c9900c611`，对应本机提交 `d68e11c11bd854c436ee45f29eeddf3aae0038be`。两者的生产源码树相同；服务器配置使用公开提交。本工具使用 `chd-offline-npy-v1` 与 `stunet-ct-nnunet-bounded-v1` 合同。

```powershell
Set-Location D:\projects\project\chd-segmentation-assistant3
$py = 'D:\projects\project\chd-segmentation\.venv\Scripts\python.exe'
& $py -B run.py check --config project.local.json
& $py -u -B run.py run --config project.local.json --run-id mini-20261115-new
```

每次使用新的运行编号，避免覆盖证据。`check`只读检查配置和生产版本；`run`会执行真实计算。配置和编号可以放在参数中，不依赖启动时所在目录。

## 有界配置

- `epochs`为1或2，`steps_per_epoch`为1或2，全局`batch`固定4；累计训练尝试必须为2至4次，以验证恢复后继续更新。AMP跳步单独记录，不追加尝试。
- 验证每轮一次，只使用所选验证病例；内部测试只预测并评价一个所选冻结测试病例。
- `profile: technical`使用32³块和从所选图像尺寸得到的共同中心32³子区域，记录实际范围。这是技术检查，不能据此证明128³容量、完整体积速度或分割泛化。
- `profile: full128`使用完整体积和128³训练块；保留小样本及更新次数限制。CUDA设置FP16；该配置需具备相应容量的设备。CPU配置使用FP32。
- `device`显式为`cpu`或`cuda`。CUDA仅请求一张空闲卡，最多等待30秒，关闭共享。资源不足直接报错。
- 保留生产项目的标签、未知类别、空间处理、预训练骨干、损失与优化器实现；不复制生产源码。

## 选例与流程

冻结划分首先由生产项目的`apply_split`读取。工具按`SHA256(seed:role:case_id)`稳定排序，从主训练角色选两名独立患者，从验证和内部测试角色各选一名；检查所有冻结角色患者互斥。患者身份缺失直接拒绝，ImageCHD需要另有可靠患者映射。

来源影像、标签及权重仅引用。只对所选四例读取体积或子区域；来源摘要在前后重新核验。保存的选择记录包含种子、来源文件摘要、源角色、匿名患者摘要、几何政策、未知类别和体积尺寸。

1. `select`：解析冻结角色，选择并核验四例，生成独立输入配置。
2. `simulation`：可选调用既有仿真环境，检查虚拟CPU/RAM/GPU资源限制、竞争释放及服务器路径；使用独立工作目录。
3. `fit`：只将两例主训练病例提交给生产项目自动计划接口。
4. `prepare`：四例共同引用本次拟合计划，检查病例、原始摘要、标签和空间参数均属于所选两例训练统计。
5. `preprocess`：生产接口首次写出四例离线数组并独立核验；改变轮数、GPU声明数量及普通日志值后确认全部复用。仅删除本次独占验证病例的一份数组，确认只补做该例，其他三例文件及时间不变。数据根目录与工作根目录分别配置。
6. `train_start`：通过只读离线映射加载数据，真实加载官方预训练权重，执行一次尝试，保存完整状态。
7. `train_resume`：重新启动进程，验证模型、优化器、scaler、随机状态恢复，继续至设定小轮终点。
8. `test`：使用冻结训练计划及最新完整检查点，运行测试病例的滑窗预测、原图空间恢复及有监督类别Dice评价。
9. `cuda_weights`：可选调用既有环境的实际单卡权重加载专项，零前向、零训练更新；记录8类别适配器的权重及保护头检查。
10. `data_verify`：核验训练、恢复、测试未重新执行整例裁剪/归一化/重采样，实际训练/验证病例准确，全部离线引用均已释放。

技术自动计划在两例训练清单上拟合；四例准备通过生产项目现有的计划引用接口接入。该路径保留生产接口对技术拟合病例数量的限制。

本工具每个运行编号使用独占、可删除的短测data，完整核验后自动清理。生产正式data由助手1维护；本工具只读核验其接口与身份规则。不同折和不同统计来源的分离在合同层检查，没有另行训练其他折或Q-A模型。日期用于目录辨识，不属于数值配方。

## 日志与清理

所有工作文件和依赖缓存定向到`.runs/<run-id>/work`。监督进程记录PID及出生时间，按0.25秒采样所属进程组RSS；每阶段最多900秒。工作文件最多500个，工作占用最多4 GiB，必要日志最多20 MiB。标准输出通过有界写入保存，为失败和清理回执预留1 MiB；日志超过保留容量时明确记录截断、原始摘要及保留字节。运行前需按使用者规则取得这些中间产物的保存授权。

成功、失败和正常中断后，记录实际文件清单与占用，核对所属进程退出，清理本次工作文件。必要日志保存在`.runs/<run-id>/logs`，保留至使用者主动删除。

- `summary.json`：终态、实际命令、依赖版本、阶段墙钟、启动收尾、训练进度、更新速度及采样内存。
- `selection.json`：固定选择、角色及输入身份。
- `fit.json`、`prepare.json`、`preprocess.json`：实际拟合与处理范围。
- `train_start.json`、`train_resume.json`：保存、恢复和累计尝试/更新。
- `test.json`：单例预测、几何恢复及监督类别评价结果。
- `simulation.json`、`cuda_weights.json`、`cuda-measured.json`：虚拟资源与实际CUDA权重专项，分别标记范围。
- `data_verify.json`：消费事件、重采样次数、数据容量与活动引用核验。
- `production-events.jsonl`：生产接口阶段、供数、训练、验证、保存和预测事件。
- `production-events.raw.log`：事件末行不完整等异常发生时保留的原始诊断；摘要明确标记失败。
- `retention.json`：诊断截断及保留容量回执。
- `processes.json`、`ownership.json`、`cleanup.json`：进程归属、保留规则、删除数量与释放字节。
- 各阶段`.log`：完整标准输出和真实错误。

并行与嵌套阶段时间分别统计，不能相加作为总墙钟。首次训练、恢复启动和稳定更新分别保留；原接口未单独提供的指标明确记为未测量。RSS是采样峰值，CUDA显存以生产训练实际记录为准。

合同与异常检查：

```powershell
$env:PYTHONPATH = 'src'
& $py -B -m unittest discover -s tests -v
& $py -B tests/verify_runtime.py --config project.local.json
```

维护检查不执行训练更新；其独立工作文件同样记录并自动清理。四例短测属于用户明确批准的范围。

## 单独更新

公开远端为 `https://github.com/tianwuasd/project.git`，独立分支为 `codex/chd-segmentation-mini-workflow`；该分支以助手3工具为根目录。首次安装仅克隆这一分支，更新器核验项目身份。

```powershell
& $py -B update.py --check
& $py -B update.py --remote origin --ref codex/chd-segmentation-mini-workflow
```

检查本工具未提交修改、活动运行锁、远端项目身份、允许文件范围、普通文件类型及快进关系后，更新器只更新自身代码。生产目录及运行配置不属于更新目标；本机私有配置与`.runs`受忽略规则保护。未配置远端时执行更新会明确报错，`--check`仍可读取本地版本。

## 集中交付

私有交付规格通过 `run.py deliver --spec <规格.local.json>` 生成指定提交的真实Git源码包和JSON清单。源码成员按允许范围核验；相同提交的源码包跨日期原地复用。清单对已有输入和权重登记路径、实际字节、SHA256、服务器位置和上传状态，源大文件不复制；元数据自身字节计入新增容量。没有服务器回执时状态为未核实。

本机交付入口为 `D:/projects/back/deliverables/先心分割处理/assistant3`，含独立服务器配置、操作说明与验收报告。私有清单、患者映射、权重及运行记录不进入公开分支。源码可由独立分支取得，源码ZIP用于可选归档。

## 检查边界

本机检查记录与实测结果保存在`.runs`；源码分支包含功能和可重复入口。128³完整体积、Linux多卡及服务器速度需分别取得真实运行证据。没有相应证据时不能标记为通过。
