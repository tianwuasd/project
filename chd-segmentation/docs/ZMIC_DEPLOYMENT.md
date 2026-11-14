# zmic 项目代码同步

根目录为 `/data4/zhangruichen`。Git 仓库固定在 `code/git`，采用部分克隆，只下载和写入 `chd-segmentation/`。code 下和 Git 仓库内部均允许存在其他项目；更新器保留它们的工作文件、未知文件及现有稀疏配置。服务器操作由用户执行，实际状态以执行回执为准。

| 职责 | 位置 |
| --- | --- |
| Git 仓库 | `code/git`，官方 origin 为 `https://github.com/tianwuasd/project.git` |
| 代码、入口、默认配置、文档 | `code/git/chd-segmentation` |
| 私有配置、清单、计划、预处理描述、检查点 | `course/chd-segmentation` |
| 更新配置与最近一次成功回执 | `course/chd-segmentation/update_network.json`、`updates/latest.json` |
| 最终掩膜和评价 | `result/chd-segmentation/<dataset>/<fold>/<run_id>` |
| 项目环境的计划位置 | `env/chd-segmentation`，需要核对实际解释器 |
| 原始数据 | `data/<实际数据集目录>`，只读 |

## 更新入口

Python 3.11+ 和系统 Git 即可执行更新，不依赖训练环境。在服务器执行：

```bash
python3 -B /data4/zhangruichen/code/git/chd-segmentation/update.py
```

仅查询当前和远端提交：

```bash
python3 -B /data4/zhangruichen/code/git/chd-segmentation/update.py --check
```

更新开始时固定 main 的完整提交号。只下载本项目缺失的文件对象；语法和快进关系检查通过后，通过候选 Git 索引和限定路径的检出更新本项目。HEAD、origin/main 和核对时的远端 main 必须一致，本项目目录保持干净，实际代码逐文件核对 Git blob。其他项目的工作文件不会被下载、检出、覆盖或删除；必要的提交、索引及分支元数据属于整个仓库。其他项目可以保留本地工作修改和不冲突的暂存内容，因此整个仓库的 status 允许显示其他项目的差异。

不使用 reset 或自动 stash。本项目的本地修改、未跟踪文件、文件类型异常，以及分支分叉、缺少入口或来源变化会阻止更新。其他项目的暂存内容若与目标提交冲突，Git 会停止操作并保留原索引和文件。网络在文件更新前失败时保留原工作目录；检出或指针写入阶段失败会报告错误，不能将其报告为同步成功，也不会自动恢复文件。候选索引与本次独占的索引锁在退出时清理，更新回执只保存一份最新成功记录。

索引快进采用 Git 的 [read-tree](https://git-scm.com/docs/git-read-tree)，实际文件写入通过 [checkout-index](https://git-scm.com/docs/git-checkout-index) 的显式路径列表限定在本项目；不执行全仓库合并检出或重新设置共享稀疏范围。

## 首次切换

使用本次交付的 `服务器代码部署_20261113.sh`。它取得固定版本的独立更新文件并执行 `--remove-legacy`，沿用已有 `code/git`，支持未初始化索引的 no-checkout 克隆，并只检出本项目。其他项目已存在时原样保留。

`--remove-legacy` 只处理 `code/chd-segmentation`：先检查登记信息、每份源码清单、根入口及逐文件摘要；同步成功后删除已核验的旧项目代码目录。任何未知文件、本地修改、链接或活动任务都会阻止删除。数据、环境、过程产物、训练检查点及最终结果均保留。重复执行时，已不存在的旧目录无需再次处理。

## 镜像配置

公开仓库默认并行探测 ghfast.top、gh-proxy.com 和 GitHub。候选必须支持部分下载、提交匹配，优先选择响应较快的可用镜像。GitHub 可达时以官方提交核对候选；官方不可达时要求可用镜像引用一致，回执标注 `mirror_agreement`，该状态不能代替官方实时核验。

首次更新在过程目录生成可手动修改的 `update_network.json`：

```json
{"mirrors": ["https://ghfast.top", "https://gh-proxy.com"], "probe_timeout_seconds": 10}
```

`--mirror auto` 使用配置；`--mirror HTTPS前缀` 指定镜像，可重复传入；`--mirror direct` 使用官方源。`--sha 完整提交` 固定目标，`--repo`、`--runtime` 可以手动更改目录。镜像请求禁用认证头、cookie 和凭据助手，不修改 origin、全局 Git 或系统代理。个别镜像探测失败按用户要求继续检查其他候选；下载或内容验证失败直接终止。

## 运行与必要保护

统一从固定代码目录使用 `run.py` 或 `smoke.py` 启动任务。两个入口在导入业务模块前持有共享代码锁；更新需要独占代码锁。同时检查旧入口的过程目录锁和训练锁，保持活动任务运行，不在任务期间修改代码。直接调用内部函数或绕过入口的外部脚本不在该锁的保护范围内。

环境和代码分别维护。先核对实际解释器，再执行：

```bash
/data4/zhangruichen/env/chd-segmentation/bin/python -B /data4/zhangruichen/code/git/chd-segmentation/setup_env.py --check
python3 -B /data4/zhangruichen/code/git/chd-segmentation/run.py --code-info
/data4/zhangruichen/env/chd-segmentation/bin/python -B /data4/zhangruichen/code/git/chd-segmentation/run.py --version
```

`--code-info` 仅使用标准库，显示当前 Git HEAD、本项目的本地修改标记、实际代码目录和调用解释器；业务命令必须使用训练环境。私有实验配置填写 course 中的 work_root、run_dir 和 result 中的 result_root，data_root 指向 data 中的真实数据集。具体数据子目录、清单、拟合计划和解释器仍需按服务器现有文件核对。

本地短测验证两份真实代码的迁移、快进、重复同步、指针与文件一致性、锁和本地修改保护。它不连接 zmic，不执行病例处理或训练；服务器部署及 CUDA/DDP 仍需实际回执核验。
