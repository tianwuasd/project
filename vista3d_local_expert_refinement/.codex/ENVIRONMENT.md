# 环境、工具与恢复

## 当前机器接续（2026-10-25）

本机工作目录为 `D:/code_project/vista3d_local_expert_refinement`，从 GitHub main 提交 `25fae29f78f40d1e8ac95f3cd5e3d38b060ce87b` 下载。下面的 D:/projects/ 和 Lenovo Python 路径为来源机器的运行留痕；本轮未恢复其运行环境，也未重新调用 Academic Agent。阅读固定快照不依赖这些旧路径。

## 原始归档环境

- 根目录：`D:/projects/vista3d_local_expert_refinement`。
- 本轮只归档文档、论文和研究工具输出；没有患者 CT、模型权重或训练环境。
- `.codex` 保存项目交接，未写入全局 Codex 设置或全局长期记忆。
- 没有注册 Codex 侧边栏项目的可用工具；本地目录已经建立，可将该目录作为项目打开。

## 已实际运行的学术工具

- 应用：`D:/projects/testing_tools/academic_agent`，Academic Agent 0.4.0。
- GitHub：`https://github.com/tianwuasd/testing_tools`。
- 研究运行时提交：`922c702f37545051097f3dbeca965c8fce14ab89`。这是运行留痕，不代表以后远程最新状态。
- 原研究项目：`D:/projects/testing_tools/academic_agent/projects/vista_expert_clicks_20261025`。
- 本项目快照：`research/academic_agent_snapshot`；固定导出 `v-2026-10-25T054638-996790-0000-0128989e`。
- Python：`C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`，运行记录为 3.12.14。
- 应用本地依赖：应用目录的 `.runs/python-deps`；属于兼容范围 pip 安装，不是 uv.lock 精确复刻。
- 启动辅助：应用目录 `.runs/run_agent.py`；将 app/src 与 app/.runs/python-deps 加入 Python 搜索路径。
- 后端使用既有 Codex 登录，未设置新 API key；实际模型名称未由结果暴露，不得自行补写。
- 国外检索当时使用临时本地代理 `127.0.0.1:7897`；续跑前检查可达，不修改全局代理。

## 恢复规则

现有快照可直接阅读。数据库使用 SQLite backup API 生成一致副本，原始 artifacts 和 exports 保留相对布局，导出哈希未改写；历史 JSON 中的绝对来源路径保留用于溯源。

若继续文献研究，先复制快照到新的 `research/academic_agent_working`，再通过现有应用指定该项目目录。不要直接修改固定快照，也不要在原项目和新工作副本上同时编辑同一研究状态。应用文件与 Python 依赖没有复制进本项目；机器迁移时应按应用 README 重建环境。

可用下列只读入口检查本地快照（不启动论文检索或模型调用）：

```powershell
$env:PYTHONUTF8 = '1'
& 'C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'D:/projects/testing_tools/academic_agent/.runs/run_agent.py' --project 'D:/projects/vista3d_local_expert_refinement/research/academic_agent_snapshot' status
```

上面的可恢复性由本次归档后实际状态检查记录验证；后续仍需检查路径是否存在。不要默认运行 `resume`，它可能触发新的模型调用。

## 历史问题

- ARS：`D:/projects/project/academic_tools`，与本应用不同。早期稀疏检出和缺失入口的诊断不再适用于正确应用。
- 两次在途阅读因添加文献触发版本保护而拒绝结果；固定语料后重跑成功，预算仍计入两次未知尝试。
- Semantic Scholar 的五次 429 是检索范围缺口；没有声称系统综述或完全查重。
- Windows 很长的中文/英文文献路径需使用长路径支持；整理副本采用短文件名。控制台乱码先检查 UTF-8 解码，不据显示认定文件损坏。
