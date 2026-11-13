# zmic 项目代码同步

根目录为 `/data4/zhangruichen`。本项目采用固定 GitHub 提交、按需取得对象、独立代码版本、逐文件验证及原子生效指针。助手准备并本地验证工具；服务器状态以用户执行回执为准。

| 职责 | 位置 |
| --- | --- |
| GitHub 提交和文件对象缓存 | `code/git`，沿用用户已经克隆的仓库 |
| 独立项目入口 | `code/chd-segmentation/update.py`、`run.py` |
| 完整源码、默认配置和文档 | `code/chd-segmentation/code_versions/<完整 SHA>` |
| 安装与生效记录 | 项目根的 `installed.json`、`current.json` |
| 引导文件、下载暂存、更新日志 | `course/chd-segmentation/bootstrap`、`updates` |
| 私有配置、清单、计划、预处理描述、恢复检查点 | `course/chd-segmentation` 下对应目录 |
| 最终掩膜、结果身份和评价 | `result/chd-segmentation/<dataset>/<fold>/<run_id>` |
| 项目环境的计划位置 | `env/chd-segmentation`；解释器需核对实际清单 |
| 原始数据 | `data/<实际数据集目录>`，只读 |

Git 缓存工作区不参与发布，缓存中的 HEAD 和删除状态不作为实际运行版本依据。同步器只读取目标提交的本项目子目录，下载其缺失对象，然后在禁止隐式联网的阶段归档、核对 Git blob、编译语法并生成逐文件 SHA256 清单。其他项目的文件不会被展开、合并或清理。当前项目代码与缓存目录相互独立。

## 首次引导

使用交付文件 `服务器代码部署_20261113.sh`。它沿用 `code/git` 的官方 origin 和已有认证，在开始时固定 main 的完整提交，显式取得并核对本次已发布的 `chd-segmentation/update.py` 对象，保存在 course 的 bootstrap 目录，再用系统 Python 3.11+ 执行该独立文件。引导文件只依赖标准库和系统 Git，无需先安装训练环境。

该入口随后只下载与安装本项目完整代码。默认结果为 `installed_not_activated`，安装位置为 `code/chd-segmentation/code_versions/<SHA>`，installed.json 记录准确提交。安装完成并不证明训练环境、数据清单或模型权重已经准备好。

## 日常同步和查询

在服务器执行，入口会在开始时固定 main 的完整提交。需要手动选择时增加 `--sha 完整提交`：

```bash
python3 /data4/zhangruichen/code/chd-segmentation/update.py
python3 /data4/zhangruichen/code/chd-segmentation/update.py --check
```

main 在同步期间变化时立即停止，当前生效指针保持原值。重复安装同一提交复用已有文件。已安装源码被修改、目标含未知文件、当前任务持有代码锁或训练锁、代码路径越界、文件缺失或校验失败时停止；不会覆盖这些修改。当前生效版本只允许快进，旧不可变版本保留给原任务。

更新事件日志位于 `course/chd-segmentation/updates/events.jsonl`，单段上限约 1 MiB，保留一份轮转段；最新成功回执为同目录 latest.json。成功或失败均清理本次独占暂存。根入口采用 protocol 1，根入口代码变化时要求明确升级，避免多文件入口切换中断。

## 环境检查与激活

核对实际环境路径后，在同一条安装命令加上 `--activate --python`：

```bash
python3 /data4/zhangruichen/code/chd-segmentation/update.py --sha SHA --activate --python /data4/zhangruichen/env/chd-segmentation/bin/python
python3 /data4/zhangruichen/code/chd-segmentation/run.py --code-info
python3 /data4/zhangruichen/code/chd-segmentation/run.py --version
```

激活前使用所指定解释器运行该版本 setup_env.py --check，检查通过才原子写 current.json。解释器缺失或依赖失败时，已下载代码保留，原生效指针不变。run.py 根据 current.json 使用已验证的实际环境和不可变源码；--code-info 显示提交、实际目录、环境及版本。正常计算持有共享代码锁，更新器不能在任务运行时切换版本。

正式分阶段命令通过受管 run.py 执行，私有配置位于 course 中，训练与预测的数据及数学配方遵循该版本配置。本次 CPU 流程的 CUDA/DDP 与服务器真实执行仍未验收。

## 数据与私有配置

私有配置填写 `work_root: /data4/zhangruichen/course/chd-segmentation`、`run_dir: <work_root>/runs/<run_id>` 和 `result_root: /data4/zhangruichen/result/chd-segmentation`。清单的 data_root 指向 data 下实际病例目录。原始数据、标签、划分、预训练权重、环境与运行日志不进入代码发布。

两个数据集的具体子目录、私有清单、既有拟合计划和实际可用解释器仍需核对。旧任务保持自身的代码与原路径，更新器不迁移或重新训练旧模型，不启用 back 中停用的预处理。
