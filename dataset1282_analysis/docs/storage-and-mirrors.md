# 服务器目录与代码更新

服务器原始数据保留在 `/data_nas/zhangruichen/dataset1282_CHD_CT_back`。全部代码、环境、缓存、预处理产物、实验结果、训练记录和检查点使用 `/data5/zhougaowei/zhangruichen_workspace`，默认工作目录为其下的 `dataset1282_work`。`project.local.json.example` 给出了机器配置。已有 NAS 预处理目录保持可读，所有新输出拒绝 `/data_nas`，解析符号链接后再次检查。输入输出目录还须彼此独立。

## 迁移已有进程目录

在服务器停止使用旧目录的全部任务，包括 SAM 训练；保持任务停止直至迁移完成。使用同一项目 home 调用迁移工具。工具检查当前用户拥有源目录及所有条目，检查所有 `operation.lock`，拒绝链接和特殊文件，并验证每一个文件的大小及 SHA256。工具不通过 SSH 连接服务器。

在已部署工程目录运行，使用它的 Python 环境：

```bash
source /data5/zhougaowei/zhangruichen_workspace/dataset1282_runtime/training-env.sh
export PYTHONPATH="$PWD/src"
"$PY" -m dataset1282_analysis.storage_migration plan \
  --source /data_nas/zhangruichen/dataset1282_preprocessing \
  --destination /data5/zhougaowei/zhangruichen_workspace/dataset1282_work \
  --plan-file /data5/zhougaowei/zhangruichen_workspace/storage-plan.json
"$PY" -m dataset1282_analysis.storage_migration apply \
  --plan-file /data5/zhougaowei/zhangruichen_workspace/storage-plan.json \
  --home "$PWD" --move
```

目标目录和计划文件必须尚未存在。上面的 `--move` 命令在验证复制结果后移除旧过程目录，使 NAS 只保留数据集；删除仅允许精确路径 `/data_nas/zhangruichen/dataset1282_preprocessing`，复制验证及再次检查活动锁通过后才执行。需要只复制时省略 `--move`。失败保留已有目标现场，检查后选择新的目标并重新生成计划；工具不会覆盖已有目标。

原始 `run.json`、prepared 请求和阶段参数按字节保留。目标目录中的 `.storage-relocation.json` 以及旁边的 `dataset1282_work.relocation.json` 记录精确旧新前缀、完整大小及哈希清单和原合同哈希。`load_relocation(path)` 验证原合同，`canonical_path(path)` 为续跑身份比较映射新路径到原路径。可变检查点可以继续训练；原合同更改会直接终止检查。迁移清单验证证明复制时内容一致，不能替代各训练流程已有的输入身份检查。

## 公开 GitHub 镜像

固定官方仓库为 `tianwuasd/project`，当前已公开。镜像固定为 `gh-proxy`（默认）及 `gitclone`：

```bash
python update.py --check --mirror
python update.py --mirror gh-proxy
python update.py --mirror gitclone
```

工具通过官方 GitHub 小型 API 确认仓库公开及 main 完整提交 SHA。镜像分支必须精确匹配该 SHA。官方接口在服务器无法访问时，先从可访问官方仓库的电脑获取完整 SHA，再在服务器提供该值：

```bash
python update.py --mirror gh-proxy --expected-commit <官方确认的40位SHA>
```

此参数表示调用者已通过官方渠道确认 SHA。工具拒绝镜像旧提交，不会猜测可用版本。镜像访问关闭凭据助手、交互认证、全局及系统 Git 配置、额外认证请求头和 HTTP 重定向；缓存中出现未允许的本地配置时终止。`--proxy` 仍只设置本次进程代理。

下载前使用 Git capability 检查 partial clone 支持；不支持 filter 的服务终止。下载仅获取浅层提交与 tree，再读取 `dataset1282_analysis` 的允许文件，逐文件校验现有 release manifest，完整 monorepo blob 不参与下载。更新收据记录官方 SHA 和所用镜像。依赖检查、活动任务锁及原子版本激活继续生效。

已有 Git worktree 中直接调用 `scripts/` 的任务，先停止任务，在该 worktree 执行下面的镜像更新，再启动新进程：

```bash
git pull --ff-only https://gh-proxy.com/https://github.com/tianwuasd/project.git main
git rev-parse HEAD
```

将显示的 SHA 与官方发布的 SHA 核对。`update.py` 管理的版本激活只供经过 `bootstrap.py` 的入口使用，直接调用脚本使用所在 worktree 的源码。

## 产物保留

短测的 synthetic、quick、capacity、预处理验收、预测验收、GPU 检查及 SAM steps 测试使用专属临时目录；结束或发生异常后删除生成文件，仅保留 `.log`。参数、输入摘要、容量报告、关键数值和异常写入日志。正式 `xu_trial.sh` 的一至两轮训练可以继续原计划，属于真实训练，保留最新状态。

训练各网络覆盖保存一个最新完整检查点，包含恢复所需的模型、优化器和已完成轮次，Xu 依据已完成轮次与固定种子恢复随机顺序；开发集选出的模型单独保留，用于预测。blood 的新最终模型引用 encoder/sequence 已选权重，避免再次保存相同张量；旧单文件模型仍可用于预测和续跑。正式训练与验证参数、选择规则和预测网格保持原设置。

每个输入数组缓存默认上限 8 GiB，可用 `XU_INPUT_CACHE_MAX_BYTES` 指定。达到上限时不再持久保存该项，按相同数组继续计算。阶段完整提交后释放输入缓存；融合分支 `.npy` 数组校验后写入释放记录并清除，续跑根据原摘要和释放记录验证，原网格预测、评价和模型保留。流水线完成后释放共享预测缓存。

预处理的规范化 CT、标签、空间元数据和计划继续保留，分别供 Xu、STU-Net、SAM 和原网格评价使用；原始数据保持只读。日志、收据、合同、必要结果和最新可续跑状态保留在 workspace。迁移仅改变位置，不改写旧运行合同。
