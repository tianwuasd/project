# 追加后续步骤与上传结果（SSH + Git LFS 0.22.0）

网络配置及旧版首次升级见 [代理与私有仓库](github-network.md)。`c` 保存原 runtime 的网络设置，`u` 使用选定的 Git 通道，`w` 上传代码，`v` 使用同一 SSH 密钥认证并通过 Git LFS 上传产物。旧包重试读取原 runtime 新设置，不必重新打包。

先更新代码并重新打开 `bash start_server.sh`。旧 workflow 不必搬迁或重跑；面板 `o` 继续读取已有结果，`r` 可给同一个任务补上后续步骤。

## 已完成任务继续运行

在面板选 `r`，选择原任务，选择并输入尚未加入的步骤编号或名称，例如 `test,predict,evaluate`。成功步骤会显示“保留已完成步骤”；新步骤使用原缓存、模型与名称，在原目录追加输出。面板 `o` 会同时显示原结果和新增结果。

```bash
bash start_server.sh --resume-job /完整路径/原任务_workflow --append-tasks test,predict,evaluate
```

- 原计划已有的步骤不能再次追加。失败或中断的步骤仍按原恢复策略继续；成功步骤不会重跑。
- 追加不从另一个最近任务读取缓存、模型或配置；未执行的诊断功能可能仍需要输入其原始标签来源；CLI 可同时给出首次需要的 `--diagnosis-source`、`--diagnosis-labels` 等参数，已有绑定不能改换。
- 已有成功产物或来源发生变化，或旧版缺少校验快照时，会阻止自动续跑并保留旧文件。结果查询仍然可用。
- 不能在绑定缓存的旧任务中追加预处理来更换数据。不同数据或训练参数属于新实验，请使用 `n`。
- 新提交前归档原 `plan.json`、状态与回执至 `attempts/`；同一 runtime 的锁和提交前复核防止两个面板覆盖计划。

## 更新线路

私有仓库只尝试选定官方 SSH/HTTPS，不回退公共镜像。公开模式下，面板 `u` 按配置线路逐个尝试。首个通过提交校验且可快进的候选即停止，包括与本地相同的版本；只有失败/无效候选才继续。若 GitHub 官方分支查询成功，仍须与官方提交一致。官方不可达时明确提示无法确认最新版本，不遍历剩余镜像比较。

## 上传日志和模型权重

面板选 `v` → 选择完成的任务 → 查看文件清单和总大小 → 确认发布。上传目的地是同一仓库 [tianwuasd/project](https://github.com/tianwuasd/project) 的独立 `chd-artifacts/<package_id>` 分支，与 main 的程序代码分开保存。

包括运行日志、训练配置/超参数、指标、模型清单，以及训练好的最佳权重和最后检查点。明确引用的外部模型目录也会收集。原始影像、缓存、预测体积、原始诊断表、环境和凭据不在上传范围内；以实际预览清单为准。

先准备包，查看清单：

```bash
bash start_server.sh --upload-job /完整路径/原任务_workflow --non-interactive
```

它会显示实际的重试命令，例如：

```bash
bash start_server.sh --upload-package /实际runtime/uploads/实际包目录 --yes-upload
```

也可明确一次完成准备与上传：

```bash
bash start_server.sh --upload-job /完整路径/原任务_workflow --yes-upload --non-interactive
```

服务器需要 Git、Git LFS，以及 `c` 中配置的 `github-ssh` 写密钥；无需 gh 登录。认证只访问官方地址，不把凭据交给公共下载镜像。配置方法见 [网络设置](github-network.md)。

上传过程：

1. 完成且无存活进程的任务持锁制作稳定快照；完整读取获准文件、计算 SHA256，流式生成最大 1 GiB 的 tar 分卷。清单包含来源路径，日志/配置内容按文件原样保存，请审阅实际预览。
2. 预览确认后，在包目录建立独立 Git 工作目录。提交只含清单、说明、LFS 属性及分卷指针；实际分卷放 LFS 对象存储。额外需要约一份包大小的本地 LFS 缓存空间。
3. 先上传 LFS 对象，再复核包和提交，最后推送独立产物分支。失败或超时不创建新包：原命令重试同一提交；已存在且不同的同名远端分支会被拒绝，不覆盖、不强推。
4. 远端 ref 与本地已核验提交一致后写成功回执。该检查证明 LFS 命令成功且指针/清单提交已推送；不等于又从 GitHub 完整下载所有大文件验证。

包和清单位于原 `<runtime>/uploads/<package_id>/`。`git-publication/` 是该包的独立 Git/LFS 暂存目录，保留它供重试。旧包不重打包；旧 `release_tag` 保留兼容，旧 Release 不会被修改。

成功后面板 `o` 显示产物分支链接；workflow 根目录 `uploads/` 保存回执，不改旧步骤输出。代码工作区保持干净，`u` 不会检出产物分支。`w` 只上传代码到 main，不用于上传权重。

下载时，使用同一 SSH 认证与代理设置，选择回执中的实际分支，并关闭克隆时自动展开，随后显式获取 LFS 文件：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch --branch 'chd-artifacts/实际包ID' \
  ssh://git@ssh.github.com:443/tianwuasd/project.git /新的产物下载目录
cd /新的产物下载目录
git lfs pull
```

上述手动 Git 命令需在当前终端显式配置 `GIT_SSH_COMMAND` 和 HTTP 代理；面板设置不会修改终端全局环境。按照 `manifest.json` 对每个分卷核对 size/SHA256，按编号合并 `artifacts.tar.part*` 后，在新的空目录解包；保留原任务，不覆盖旧输出。不要将产物分支合并到 main。

单卷 1 GiB 低于 GitHub 常见套餐的 LFS 单文件上限，但存储、下载带宽和账单限制仍适用；额度错误时保留包并停止，不自动购买或修改额度。[GitHub LFS 说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)

本地测试使用真实的临时 Git/LFS 仓库和合成文件，验证上传、克隆及字节恢复；没有上传真实模型或患者数据。Linux 进程回收、zmic44 权限/网络/额度与真实吞吐需服务器验收。

2026-10-25 的 Windows / Python 3.12 验证：相关工程回归 143 项通过、1 项因符号链接权限跳过；另 1 项完整模型入口测试因未安装 torch/scipy/skimage/networkx/sklearn 未通过，不能算全套通过。最后代码上传目录保护与 CLI 集成复核 18 项通过。Ruff 通过；未在目标 Linux 执行，未重复触发此前因 GitHub 账户计费/额度限制未启动的 CI。
