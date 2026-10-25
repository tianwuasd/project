# GitHub 代理、私有仓库更新与结果上传（0.22.0）

面板新增 `c GitHub 网络设置`；`u` 更新代码、`w` 上传已提交代码、`v` 上传完成任务的产物分别读取其原 runtime 的 `network.json`。配置只保存代理地址、Git 线路和密钥路径，不保存订阅、密码、令牌或密钥内容。保存设置不代表连通性验证通过。

## 已升级后的配置

zmic44 的本机 HTTP 代理地址为 `http://127.0.0.1:17891`。这是服务器上的地址；Windows 本机的 7897 不是它。当前私有仓库可选择 `github-ssh`，沿用用户已有的部署密钥：

```bash
cd /data5/zhougaowei/zhangruichen_workspace/chd_ct_code/project/chd-ct-reproduction
bash start_server.sh \
  --runtime /data5/zhougaowei/zhangruichen_workspace/chd_ct_runtime \
  --configure-network \
  --github-proxy http://127.0.0.1:17891 \
  --github-transport github-ssh \
  --github-ssh-key "$HOME/.ssh/project_readonly_ed25519"
bash start_server.sh --show-network
bash start_server.sh
```

默认服务器配置与上述 runtime 相同；使用其他 runtime 时，查看、启动和重试命令也必须指定它。面板 `c` 可以修改这些设置；输入 `b/q` 或 Ctrl+C 取消不写入半份配置。CLI `--github-proxy none` 清除显式代理，之后沿用父进程网络环境，不保证强制直连。

| 线路 | 代码更新认证 | 失败后行为 |
|---|---|---|
| `github-ssh` | `ssh.github.com:443`、指定密钥、严格主机校验；HTTP 代理通过 OpenBSD `nc -X connect -x` | 停止并保留记录，不切公共镜像 |
| `github-https` | 官方 HTTPS，沿用已有 Git 认证助手 | 停止并保留记录，不切公共镜像 |
| `public-mirrors` | 仅公开仓库，抑制凭据；未配置文件时保留旧默认 | 有界镜像回退；显式代理时先试原站 |

SSH 模式需要兼容上述参数的 OpenBSD nc；只找到名为 nc 的程序不证明参数兼容。密钥需存在、可非交互使用，GitHub 主机指纹需提前核验并记录。SSH 配置不会替用户接受未知主机或弹出密钥密码输入。官方地址被 Git `insteadOf` 改写时拒绝更新，不向镜像发送认证。

`u` 仍检查工作区、分支、任务锁与快进关系，更新前备份，保留原 origin 和 sparse-checkout。它更新整个 monorepo 的提交；其他已检出的子目录也可能随提交变化，不能称为只更新 CT。已有稀疏检出范围保持不变。网络设置不触发拉取、推送或训练。

## 旧面板首次升级

0.20.0 及更早的 `u` 不支持此网络配置，也不向私有仓库提供认证。先确认共享这份代码的所有任务均已结束；仅退出面板不会结束后台任务。下面的一次性命令持默认 runtime 的任务锁、拒绝脏工作区、备份当前 CT 代码，再从官方 SSH 快进更新。其他 runtime 的任务应先另外确认停止。需要 `nc` 和 `flock` 已安装，且已完成官方 SSH 主机指纹核验。

```bash
(
set -e
cd /data5/zhougaowei/zhangruichen_workspace/chd_ct_code/project
runtime=/data5/zhougaowei/zhangruichen_workspace/chd_ct_runtime
mkdir -p "$runtime"
exec 9>"$runtime/.job.lock"
flock -n 9 || { echo '任务仍在运行，请先在面板确认退出。'; exit 1; }
[ "$(git branch --show-current)" = main ] || { echo '当前不是 main。'; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo '有本地修改，请先保存处理。'; exit 1; }
command -v nc >/dev/null
[ -f "$HOME/.ssh/project_readonly_ed25519" ]
url=ssh://git@ssh.github.com:443/tianwuasd/project.git
[ "$(git ls-remote --get-url "$url")" = "$url" ] || { echo '存在 Git URL 重写，请先检查。'; exit 1; }
export GIT_SSH_COMMAND='ssh -F /dev/null -i ~/.ssh/project_readonly_ed25519 -p 443 -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o "ProxyCommand=nc -X connect -x 127.0.0.1:17891 %h %p"'
export GIT_SSH_VARIANT=ssh
backup=$(mktemp -d "$runtime/upgrade-0.22-XXXXXXXX")
git rev-parse HEAD >"$backup/before.txt"
git archive --format=tar --output="$backup/ct-before.tar" HEAD chd-ct-reproduction
timeout 120 git fetch --no-tags --no-recurse-submodules "$url" main
git merge --ff-only --no-overwrite-ignore --no-edit FETCH_HEAD
git rev-parse HEAD
echo "备份位于 $backup"
)
```

失败保留备份和旧配置，不用 reset 或删除锁绕过。升级后执行上节配置并重新打开面板。上述命令只在括号内设置 SSH 环境，不改全局 Git 或 shell 配置。此文档提供操作步骤，不代表已在 zmic44 执行。

## 通过同一 SSH 密钥上传

服务器无需 gh 或 API token。`u` 使用 `github-ssh` 读取；`w` 和 `v` 要求同一模式及有写权限的密钥。密钥文件名中的 readonly 不决定实际权限。只读认证成功不等于有写权限。

| 操作 | 目标 | 依赖 |
|---|---|---|
| `u` 更新 | 官方仓库 main | Git、OpenSSH；显式代理时需 OpenBSD nc |
| `w` / `--push-code` | main 中已提交的 CT 代码 | 同一 SSH 密钥写权限；不自动 add/commit，不强推 |
| `v` / `--upload-job` | `chd-artifacts/<package_id>` 独立分支 | Git LFS；同一 SSH 密钥认证；LFS 存储/带宽可用 |

代码上传先显示提交及文件清单，工作区须干净，远端须是本地祖先；只接受 CT 范围内的文本源文件，单个新 blob 不超过 2 MiB，不接受合并提交或历史中混入其他项目/运行产物。需要特殊代码发布时人工核对 Git 提交。日志和权重使用 `v`。

```bash
git --version
git lfs version
bash start_server.sh --push-code --non-interactive
# 查看预览后，确实要上传时：
bash start_server.sh --push-code --yes-upload --non-interactive
```

Git LFS 通常经 HTTPS 传数据，通过 SSH 密钥取得授权；不等于把大型权重直接塞进普通 Git 对象。代理作用于本次 Git/LFS 子进程。Git LFS 不存在或额度不足时停止，保留原包和已有缓存；不会改账单或退回普通 Git 大文件。[Git LFS 官方认证说明](https://github.com/git-lfs/git-lfs/blob/main/docs/api/server-discovery.md)

旧上传包继续用原来的 `--upload-package` 命令，读取清单中原 runtime 的最新设置。旧清单的 `release_tag` 仅保留兼容字段；新版不会续传旧 Release，而是将该原包上传为独立产物分支，旧 Release 保留原状。详细操作及下载见 [结果发布](results-publishing.md)。

## 作用范围与排查

- 代理只传给本次 Git/LFS 子进程，覆盖其代理与绕过变量，不写全局代理、DNS、TUN、训练环境或其他用户配置。代理进程仍使用共享机器的带宽；回环监听不是按 Linux 账号隔离。
- `curl --proxy ... https://github.com` 返回成功，只证明该次 HTTPS 请求；SSH、Git 拉取、代码推送和 LFS 上传各需单独验证。不要用公开主页的 200 状态代替私有仓库验收。
- 目前 nohup 启动的 Mihomo 通常可在 SSH 断开后继续；服务器重启、崩溃或管理员终止后不保证恢复。订阅的 86400 秒更新只在进程运行且订阅可达时尝试，不会自动升级 Mihomo 或项目代码。
- 坏 `network.json` 不会被覆盖。先在原 runtime 将它改名备份，再通过 `c` 或配置命令重新建立；文件不存在时是公开镜像默认模式，私有仓库需重新选择官方线路。
- 更新记录位于 `<runtime>/updates/`；上传包在原 `<runtime>/uploads/`；代码上传回执在 `<runtime>/code-uploads/`。配置成功、本地合成测试、GitHub 发布和服务器部署分别记录。

此改动保持原 CPU/GPU 调度、科学配置、缓存、权重和历史任务不变。资源优化仍按 [服务器说明](server-start.md#非训练-cpu-分配与测试预测多-gpu0130) 的原流程执行。
