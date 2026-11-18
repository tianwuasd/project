# 先心代码助手3

工具项目名`chd-mini-workflow`，通过外部生产run.py执行真实技术步骤。完整main仓库中的目录为`chd-mini-workflow/`，生产项目位于相邻的`chd-segmentation/`。独立工具分支为`codex/chd-assistant3-imagechd-acceptance-20261117`。

2026-11-18真实阶段3使用生产`7d86dff7469d7e5a24e3c6ca65b31d5a7993959d`、工具`25044fc6940d6f628796a414009f4f7038fd156d`：STU/Q两例首次预测、同目录接续复用、原图空间几何和新评价通过，零新增训练。结果仅覆盖技术子区域。Med官方CPU运行因内存不足结束、零更新，训练恢复及训练后预测未验证，整体状态为`INCOMPLETE`。工具`d6621191671066f495982fc5fba902ba9c99a71f`增加明确批准的Q定位器次数合同，独立控制检查通过。main合入的入口检查不构成新模型运行证据。

默认1–2例技术流程，候选ct_1001训练、ct_1012验证，无独立测试集合。无追加批准时每阶段累计最多2次、总计10次；本轮匹配批准文件允许Q base与localizer各3次，其余各2次、总计12次。首次、恢复、失败和跳步共同计数。患者独立性、正式Q四组、CUDA/NCCL及正式128³容量分别需要真实证据。

## 入口

使用已有生产Python环境，运行`run.py --help`查看命令。

完整main仓库从仓库根目录运行：

```powershell
python -B .\chd-mini-workflow\run.py --help
python -B .\chd-mini-workflow\run.py review --help
```

run.py按自身目录加载`src/chd_mini`，不依赖当前工作目录。私有配置中的`engine_root`可指向同一完整仓库的`chd-segmentation/`，`engine_commit`填写实际main完整SHA；实际执行前仍需相应版本审核和运行许可。已有7d86阶段3报告保留原版本记录。

- `check --config PATH`：只读检查既有schema 2配置和生产版本，保留旧配置读取。
- `inspect --config PATH`：只读检查schema 3步骤、角色和生产版本。
- `run --config PATH --run-id ID --approval PATH`：保留参数入口，本轮明确拒绝独立训练，训练由助手2统一执行。
- `review --handoff PATH [--execute]`：生成已交接模型的预测步骤，明确放行后执行阶段3零额外训练验收。
- `attempt --ledger PATH --stage STAGE`：生产训练输入构建前领取累计尝试；完成后用`--number NUMBER --outcome OUTCOME`登记结果。
- `deliver --spec PATH`：已有源码交付入口，使用前单独核对私有规格与发布范围。

配置、累计尝试协议、实际命令要求、产物和清理约定见[集成说明](docs/imagechd-integration.md)。公开`configs/example.json`保留schema 2配置形式供兼容读取；新三模型命令模板在最终生产CLI确定后提供。不能用旧四例流程验证新功能。

## 所有权与运行证据

公共加速由助手1维护，模型角色、训练、早停、best和空间还原由助手2维护。本工具维护编排、累计记录、日志和清理，不复制生产算法。生产smoke在实际调用方迁移验收前保留。

私有配置及病例映射不提交。原始数据、权重、环境和共享数组原地引用；本轮工作与报告目录通过私有配置指定，产物容量须由秘书集中批准。生产版本只记录完整SHA和局部修改状态，不扫描源码和大数据摘要。实际生产入口继续负责运行代码锁。

必要报告包含steps、attempts、summary、ownership和cleanup。阶段墙钟单位为秒，容量单位为字节；资源计时与显存从公共生产记录获取，未采集的数据标记未测量。阶段边界容量盘点不保证瞬时磁盘峰值，运行前须核算实际同时存活文件。

真实生产恢复、best预测、跨spacing、清理和完整路线须使用已审核版本及明确批准的产物范围。本轮已获准将完整工具源码放入main的`chd-mini-workflow/`，保留其他项目。

## 更新方式

独立分支的`update.py`要求工具自身目录为Git根，只更新对应工具分支，继续兼容独立使用。不要使用`update.py --ref main`：完整main包含多个项目，不能更新到独立工具根目录。在完整main checkout中，通过仓库自身的更新流程更新代码，再运行`chd-mini-workflow/run.py`；前缀内的独立更新器会拒绝该目录作为Git根。本次不扩展更新器。
