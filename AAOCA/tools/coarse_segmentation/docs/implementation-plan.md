# AAOCA 粗分割 Implementation Plan

**Goal:** 交付现有 AAOCA 项目中的独立粗分割工具和服务器网页面板。
**Architecture:** 输入清单、几何检查、推理调度、合并与 QC、HTTP 面板分别成模块。面板使用与命令行相同的运行入口。
**Tech Stack:** Python 3.11+，NumPy，NiBabel；实际推理额外安装 TotalSegmentator 2.18.0；网页无外部依赖。
**Spec:** [design.md](design.md)
**Execution:** 当前会话直接实现并自查；用户已授权生成代码及推送。

## Global Constraints

- 只发布源码、合成测试和公开说明；运行文件在源码树外。
- 不猜测 HU/间距，不改输入，不进行真实患者推理或 GPU 资源分配。
- 11 类合并图不代表经过人工审核的真值；保留二值掩膜与重叠记录。
- Windows 本地验证；Linux 服务器运行说明必须标明未经远程实测。

## Review Focus

- 损坏或改动后的同名输出不可复用。
- 原始与预测网格不一致不可只复制空间头。
- HTTP 请求不可构造任意命令、路径或终止他人任务。
- 进程中断不得留下“完成”状态，不能静默删除失效锁。
- 同名病例、未知类别、空选择及非法设备参数必须拒绝。

## Tasks

- [x] `schema.py`、`images.py`、`cli.py`：CSV/研究清单和几何预检；测试未知单位、重复 ID、确认缺失、qform/sform 冲突。
- [x] `pipeline.py`：参数化调用、逐例隔离、锁、哈希复用、11 类合并；用合成推理替身验证失败与恢复。
- [x] `panel.py`、`web/`：独立鉴权面板、固定数据边界、病例选择、进度/日志、GPU 状态、停止自有任务；HTTP 集成测试及页面检查。
- [x] `README.md`、`docs/server.md`：安装、许可、SSH 转发、批处理和 Slicer 复核，更新 AAOCA 总入口。
- [x] 验证命令行、全部新测试、公开文件白名单，准备仅包含本次 AAOCA 变更的提交。发布状态以 Git 提交与远端核对为准。
