# 先心代码助手3

独立工具分支`codex/chd-assistant3-imagechd-acceptance-20261117`，通过外部生产run.py执行真实技术步骤。当前源码正在接入2026-11-17公共层与模型层；三模型完整功能尚未通过运行验收。

默认1–2例技术流程，候选ct_1001训练、ct_1012验证，无独立测试集合。STU、Q base/localizer/local和Med sequence每阶段累计最多2次尝试，五阶段最多10次，首次、恢复、失败和跳步共同计数。患者独立性、正式Q四组、CUDA/NCCL及正式128³容量分别需要真实证据。

## 入口

使用已有生产Python环境，运行`run.py --help`查看命令。

- `check --config PATH`：只读检查既有schema 2配置和生产版本，保留旧配置读取。
- `inspect --config PATH`：只读检查schema 3步骤、角色和生产版本。
- `run --config PATH --run-id ID --approval PATH`：执行对应明确许可的schema 3技术步骤。
- `attempt --ledger PATH --stage STAGE`：生产训练输入构建前领取累计尝试；完成后用`--number NUMBER --outcome OUTCOME`登记结果。
- `deliver --spec PATH`：已有源码交付入口，使用前单独核对私有规格与发布范围。

配置、累计尝试协议、实际命令要求、产物和清理约定见[集成说明](docs/imagechd-integration.md)。公开`configs/example.json`保留schema 2配置形式供兼容读取；新三模型命令模板在最终生产CLI确定后提供。不能用旧四例流程验证新功能。

## 所有权与运行证据

公共加速由助手1维护，模型角色、训练、早停、best和空间还原由助手2维护。本工具维护编排、累计记录、日志和清理，不复制生产算法。生产smoke在实际调用方迁移验收前保留。

私有配置及病例映射不提交。原始数据、权重、环境和共享数组原地引用；本轮工作与报告目录通过私有配置指定，产物容量须由秘书集中批准。生产版本只记录完整SHA和局部修改状态，不扫描源码和大数据摘要。实际生产入口继续负责运行代码锁。

必要报告包含steps、attempts、summary、ownership和cleanup。阶段墙钟单位为秒，容量单位为字节；资源计时与显存从公共生产记录获取，未采集的数据标记未测量。阶段边界容量盘点不保证瞬时磁盘峰值，运行前须核算实际同时存活文件。

源码编写及静态检查可以先完成。真实生产恢复、best预测、跨spacing、清理和完整路线须在上游最终发布及产物批准后由本助手执行。源码发布只使用独立工具分支。
