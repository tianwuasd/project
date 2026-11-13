# zmic 部署位置与 Git 同步

本布局依据用户确认的 2026-11-13 目录。根目录为 `/data4/zhangruichen`；助手只准备代码与命令，服务器安装情况以用户执行回执为准。

| 职责 | 本项目位置 |
| --- | --- |
| 与 GitHub 同步的 Git 仓库 | `/data4/zhangruichen/code/git` |
| 当前项目代码 | `/data4/zhangruichen/code/git/chd-segmentation` |
| 私有配置、清单、既有拟合计划 | `/data4/zhangruichen/course/chd-segmentation/configs`、`inputs` |
| 预处理描述、日志和完整恢复检查点 | `/data4/zhangruichen/course/chd-segmentation/prepared`、`runs` |
| 短测及清理记录 | `/data4/zhangruichen/course/chd-segmentation/smoke`、`smoke-reports` |
| 最终掩膜、来源记录、评价报告 | `/data4/zhangruichen/result/chd-segmentation/<dataset>/<fold>/<run_id>` |
| 独立 Python 环境的计划位置 | `/data4/zhangruichen/env/chd-segmentation` |
| 原始数据 | `/data4/zhangruichen/data/<实际数据集目录>`，只读 |

`code/git` 是仓库根，Git HEAD 表示此处真实检出的 main 提交。当前只展开 `chd-segmentation`；Git 元数据和源码一起保存在该目录。根目录之外的过程文件、结果、数据和环境不纳入代码提交。既有任务继续使用自己的原路径和版本；本说明不迁移旧恢复点，也不启用 back 中停用的旧预处理。

## 首次取得代码

首次命令适用于 `code/git` 尚不存在的情况。将 `SHA` 填为本次已经核验的完整发布提交，在服务器 Bash 执行；采用既有 GitHub 官方仓库认证。

```bash
(
set -euo pipefail
ROOT=/data4/zhangruichen
REPO="$ROOT/code/git"
SHA=填写本次完整发布提交
test -d "$ROOT/code"
test ! -e "$REPO"
git -c http.followRedirects=false clone --no-checkout --filter=blob:none --depth=1 --single-branch --no-tags --branch main https://github.com/tianwuasd/project.git "$REPO"
test "$(git -C "$REPO" rev-parse HEAD)" = "$SHA"
git -C "$REPO" -c http.followRedirects=false sparse-checkout set --no-cone '/chd-segmentation/'
test -s "$REPO/chd-segmentation/run.py"
git -C "$REPO" diff --exit-code HEAD -- chd-segmentation
printf '代码提交：%s\n项目目录：%s\n' "$SHA" "$REPO/chd-segmentation"
)
```

该命令使用 Git 的 [按需获取文件内容](https://git-scm.com/docs/git-clone) 和 [子目录检出](https://git-scm.com/docs/git-sparse-checkout)。当前模式只展开本项目路径；上层目录文件不在展开范围。遇到目标已存在、认证失败或分支已推进至其他提交时停止。当前仅完成命令语法核对，尚无服务器执行证据。

## 后续同步

仅在本项目计算已经结束、私有配置已放到 course 后更新。核对 `origin` 为 `tianwuasd/project` 官方地址、当前分支为 main、工作区干净，且 `git sparse-checkout list` 仅包含 `/chd-segmentation/`。存在本地修改、其他展开目录或未知仓库时停止，保留当前文件。

```bash
(
set -euo pipefail
REPO=/data4/zhangruichen/code/git
SHA=填写本次完整发布提交
test "$(git -C "$REPO" branch --show-current)" = main
test -z "$(git -C "$REPO" status --porcelain --untracked-files=normal)"
test "$(git -C "$REPO" sparse-checkout list)" = /chd-segmentation/
git -C "$REPO" -c http.followRedirects=false fetch --filter=blob:none --no-tags origin main
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = "$SHA"
git -C "$REPO" merge --ff-only "$SHA"
test "$(git -C "$REPO" rev-parse HEAD)" = "$SHA"
git -C "$REPO" diff --exit-code HEAD -- chd-segmentation
)
```

同步采用 Git 的 [仅快进合并](https://git-scm.com/docs/git-merge)，不自动处理本地分叉。完成后核对 Git HEAD、源码差异和实际入口位置。源文件变更会影响当前训练记录的代码身份，旧任务继续使用其已登记版本；当前没有自动跨版本续训迁移。

## 环境与私有输入

独立环境的预期解释器为 `/data4/zhangruichen/env/chd-segmentation/bin/python`，此名称是新部署位置约定，尚未证明该环境已经安装。若 env 中已有适用环境，先核对其实际解释器和依赖，不自动重建或修改它。依赖按 `pyproject.toml` 安装到项目独立环境；本次模型流程支持 CPU，CUDA/DDP 尚未接入验收。

私有配置填写 `work_root: /data4/zhangruichen/course/chd-segmentation`、`run_dir: <work_root>/runs/<run_id>` 和 `result_root: /data4/zhangruichen/result/chd-segmentation`。省略 result_root 的旧配置沿用 work_root。预测日志保存在 run_dir；结果目录中的身份摘要用于重复预测复用。

原始数据通过私有清单的 `data_root` 指向 data 下的实际子目录，病例路径可以相对该目录。两个数据集的实际子目录、私有清单、既有拟合计划、预训练权重和可用解释器路径仍需核对。配置示例中的这些输入位置是约定，未把目录截图当作文件已准备的证据。

先在独立环境执行 `python -B setup_env.py --check`，随后使用 README 的分阶段命令。首次代码命令只取得源码；正式计算使用实际环境解释器、course 中的私有配置和新的运行目录。
