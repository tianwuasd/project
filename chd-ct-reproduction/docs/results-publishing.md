# 追加后续步骤与上传结果（0.15.0）

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

面板 `u` 按配置线路逐个尝试。首个通过提交校验且可快进的候选即停止，包括与本地相同的版本；只有失败/无效候选才继续。若 GitHub 官方分支查询成功，仍须与官方提交一致。官方不可达时明确提示无法确认最新版本，不遍历剩余镜像比较。

## 上传日志和模型权重

面板选 `v` → 选择完成的任务 → 查看文件清单和总大小 → 确认发布。上传目的地是同一仓库 [tianwuasd/project 的 Releases](https://github.com/tianwuasd/project/releases)，与程序代码分开保存。

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

服务器需已安装并登录 GitHub CLI，账号拥有该仓库发布权限。认证未完成时，按照工具提示执行 `gh auth login --hostname github.com`。上传使用官方 GitHub 认证通道和服务器网络配置；公共 GitHub 下载镜像不支持安全的认证写入，不把令牌交给镜像。

上传包在原 runtime 的 `uploads/` 下，准备时会读取全部获准权重并计算哈希，需要磁盘空间与时间。分块附件支持大模型；下载后按包内说明合并解包。发布前保持 draft，失败后保留包和相同 release tag；使用打印的重试命令继续，勿重新打包产生另一个发布。

成功后面板 `o` 显示发布链接。上传回执位于 workflow 根目录 `uploads/`，不写入旧步骤输出，因此不会破坏其恢复校验。工作区保持干净，后续 `u` 不受大模型文件影响。

GitHub 限制依据：[Release 附件规则](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)、[gh release upload](https://cli.github.com/manual/gh_release_upload)。真实服务器网络、认证与上传吞吐需要在服务器验证；本地测试使用合成文件和模拟服务，不发送真实实验数据。
