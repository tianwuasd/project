# Assistant2 Offline Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Q与MedSAM2消费一次生成的兼容离线数据，并完成main发布、独立部署、资源及材料交付。

**Architecture:** 先记录当前可用版本的真实数值基线。助手1最终发布后继承其SourceReader、离线存储、资源及引用合同；Q复用STU数值处理并核验A统计，MedSAM2提供原生灰度窗配方。模型训练、预测和恢复核对具体数据身份。

**Tech Stack:** Python、NumPy NPY/NPZ、nnU-Net、官方MedSAM2、PyTorch CPU/Gloo、既有Windows服务器仿真器、GitHub代码对象接口。

**Spec:** ../specs/2026-11-15-assistant2-offline-design.md

## Global Constraints

- 冻结69289532a3da2c94bc72769f5726ca260f6980a7；工作分支codex/chd-assistant2-offline-preprocess-20261115；用户最新授权验收后整合main。
- 助手1完成测试发布后核对最终提交与公开子树，才继承数据代码；当前开发文件仅用于接口审阅。
- Q r2、A393/B197/C128/D456、A拟合和上游冻结、科学计算及随机顺序保持。
- MedSAM2官方332f30d、权重SHA、八帧512、完整编码器、原生网格与当前灰度窗规则保持。null灰度窗来自正确训练来源计划。
- 最多两例真实病例、每阶段两次更新；同时200文件/8GiB，back独占工作目录，报告总量10MiB；禁止登录服务器和自行使用GPU。
- 新增材料集中先心分割处理/assistant2；既有权重、输入包及依赖原地引用，不公开上传私有材料。

## Review Focus

- MedSAM2原生uint8数据不能进入STU/Q的标准化输入；不同灰度窗及统计来源必须拒绝混用。
- 未完成、损坏或错版本数据应在训练更新之前拒绝，普通预测不能打开标签数组。
- 日期、GPU数、轮数、日志和不同run目录不得重复生成兼容数据；实际Q-A或折统计变化必须隔离。
- Q每样本随机轨迹和官方多worker序列随机轨迹在离线读取后保持；旧在线任务使用冻结代码恢复。
- 多消费者引用、清理保护、独立分支更新和发布不能影响另一助手、原始数据及仍需恢复的任务。

### Task 1: 冻结真实数值基线

**Files:** 新建tests/verify_model_offline_real.py；验收基线JSON保存在已批准报告目录。

**Interfaces:** 基线记录case SHA、ROI、完整拟合plan、Q数组/性质摘要及MedSAM2已解析窗口/原生窗口数组摘要；不保存体素副本。新模块接口预定models.offline.preprocess_model(model_path, prepare_config_path, prepared_path, root, output) -> 完整prepared记录。

- [x] 用当前可用CaseTransform和官方数据接口的window_image，在ct_1003的32³范围生成独立数值基线；零次更新并清理。
- [x] 同一检查读取基线并调用新接口，观察新离线模块缺失的RED；保留实际错误和清理记录。

### Task 2: 继承确定公共合同

**Files:** 公共preprocessing、data、config、training、CLI及相关测试；源版本锁定说明。

**Interfaces:** 消费助手1最终公开提交对应的SourceReader、offline.preprocess、offline_store的存储、校验、只读加载及引用接口。核对实际签名后，计划中对应接口变化记入ledger；开发中实现不作为来源。保留助手2validate_resources出口、模型memory_profile及恢复设备检查。

- [x] 核对最终本地/公开提交和完整子树，读取已提交源码及其验收记录。
- [x] 通过文件编辑工具继承公共实现，逐项保留模型接口；标明允许的模型配方扩展。
- [ ] 公共STU真实有界入口通过，原模型预算和资源验证出口通过。

### Task 3: 模型配方、入口与复用

**Files:** 新建models/offline.py及models/medsam2/native.py；修改models_cli.py、MedSAM2 data/adapter/official、配置模板及必要公共配方参数。

**Interfaces:** preprocess_model使用模型配置和来源prepare配置。Q调用公共STU处理并核验A组统计；MedSAM2解析gray_window或训练计划百分位，保存uint8原生图像及int16标签。native.resolve_window(config, prepared)；native.NativeCases(prepared, with_label, ...)只读对应原生配方。CLI为models preprocess --config MODEL --prepare-config SOURCE_CFG --prepared SOURCE_JSON --root DATA --output OFFLINE_JSON。

- [x] 实现完整数据配方和只读消费者，数值摘要必须与Task1冻结基线相同；正式Q统计限定A，正式MedSAM2使用原冻结训练身份。
- [x] 测试同一配方复用、仅损坏病例重建、不同窗/统计来源隔离、不完整数据拒绝和普通无标签读取。
- [x] 测试共享引用和目录删除影响；保持数据与run身份分开，正式数据不自动删除。

### Task 4: 训练、预测与恢复

**Files:** 修改模型workflow/prediction/execution/memory及相关真实短测入口、精确兼容记录。

**Interfaces:** 训练和预测只接受兼容完整离线描述，执行期间持有数据引用。Q/MedSAM2的科学配置及检查点仍按原规则核对；新的数据版本不伪装成旧在线身份。

- [ ] 适配实际离线加载预算；保留Q整例概率、MedSAM2完整CPU训练门槛和恢复worker布局。
- [ ] Q三个阶段各一次更新后恢复到两次，实际预测/评价通过；官方MedSAM2实际两批数据和窗口数值一致，实际预测入口通过。
- [ ] 完整MedSAM2训练在主机预算不足时使用真实准入拒绝证据，不绕过内存门槛，不将数据检查表述为训练验收。
- [ ] 全套单元检查通过，原始输入摘要保持；精确兼容目标对应最终源码，未知来源继续拒绝。

### Task 5: Windows仿真、容量、材料与发布

**Files:** Windows实际监督器适配短测、服务器资源说明、集中交付清单和最终验收报告。

**Interfaces:** 只读复用D:/projects/test/common.py的实际supervise/目录采样及其正式入口；固定模型生产源码，真实CPU与虚拟资源分别记录。公开代码树以最新确定main为父树，按项目路径整合经过审阅和验收的代码，核对main既有文件和其他项目。

- [ ] 运行真实模型入口和仿真资源控制，记录分阶段时间、速度单位、RAM及磁盘采样峰值，显存未测量时明确标记。
- [ ] 提供数据/模型/检查点最终及写出峰值估算，给出数量、dtype、形状、保留份数与服务器条件时长依据。
- [ ] 更新材料新增/复用清单、实际容量、摘要、服务器路径和上传状态；不复制大文件。
- [ ] 一次新上下文整体审阅覆盖6928953以来全部分支变更；修复重要发现并运行针对性检查和单元检查。
- [ ] 本地提交并整合main；重新查询main SHA及源码子树，核对其他项目不变，不登录服务器。
- [ ] 自动清理本次副产物并核验，交付冻结/本地/公开提交、独立更新与完整运行命令、日志和清理回执。
