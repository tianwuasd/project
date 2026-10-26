# AAOCA 预处理：一次性过渡到私有仓库更新

保留原代码目录、runtime、输入输出和环境。独立粗分割由原入口启动，本次不改。下面命令在服务器执行，使用已有部署密钥和本机 17891 代理；不需要 gh 或 API token。GitHub 主机密钥须已核验并存在于 known_hosts，nc 须支持 OpenBSD CONNECT。

## 先验证读取权限

```bash
GIT_SSH_COMMAND="ssh -F /dev/null -i '$HOME/.ssh/project_readonly_ed25519' -p 443 -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o 'ProxyCommand=nc -X connect -x 127.0.0.1:17891 %h %p'" \
timeout 30 git ls-remote ssh://git@ssh.github.com:443/tianwuasd/project.git refs/heads/main
```

返回提交号和 `refs/heads/main` 证明可读取该私有仓库，不代表具有写权限。认证失败不会通过公共镜像重试；不要把私钥内容粘贴到命令、面板或聊天中。

## 已有旧代码或旧独立发布目录

先从旧面板确认预处理后台已停止；此升级器也会检查历史进程和锁。原 Git 工作区必须干净且在 main。临时下载只取 AAOCA 代码；更新器备份旧任务实际使用的代码，在原 runtime 发布新版并切换指针，不改原共享 checkout。失败保留原版本和下载目录。

```bash
cd /data5/zhougaowei/zhangruichen_workspace/aaoca_code
(
  set -eu
  export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
  export GIT_SSH_COMMAND="ssh -F /dev/null -i '$HOME/.ssh/project_readonly_ed25519' -p 443 -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o 'ProxyCommand=nc -X connect -x 127.0.0.1:17891 %h %p'"
  bootstrap_dir=$(mktemp -d /tmp/aaoca-private-upgrade.XXXXXX)
  timeout 180 git clone --filter=blob:none --no-checkout --single-branch --branch main \
    ssh://git@ssh.github.com:443/tianwuasd/project.git "$bootstrap_dir/project"
  git -C "$bootstrap_dir/project" sparse-checkout set AAOCA
  timeout 180 git -C "$bootstrap_dir/project" checkout main
  python_bin=/data5/zhougaowei/zhangruichen_workspace/aaoca_runtime/venv/bin/python
  if [ ! -x "$python_bin" ]; then python_bin=python3; fi
  "$python_bin" "$bootstrap_dir/project/AAOCA/upgrade_aaoca_server.py" \
    --checkout /data5/zhougaowei/zhangruichen_workspace/aaoca_code/project \
    --runtime /data5/zhougaowei/zhangruichen_workspace/aaoca_runtime/panel \
    --github-proxy http://127.0.0.1:17891 \
    --github-transport github-ssh \
    --github-ssh-key "$HOME/.ssh/project_readonly_ed25519"
  printf '临时下载保留于：%s\n' "$bootstrap_dir"
)
```

如果历史原来在 `~/.local/state/aaoca-server`，把上面的 `--runtime` 改成那个原路径，不另建历史。其他安装位置同样替换原 checkout/runtime；已有 `release.json` 的 checkout 必须匹配。若系统 Python 缺 psutil 而存在任务进程记录，升级会因无法确认存活而停止；请用原预处理环境，不绕过检查。

升级成功后仍用原命令：

```bash
cd /data5/zhougaowei/zhangruichen_workspace/aaoca_code/project/AAOCA
bash start_aaoca_server.sh
```

以后通过 `a` 修改网络设置，`u` 更新；不再需要临时下载。更新不升级已有 Python 环境。新工作台中 `c` 是预处理配置，`m` 是 PyPI 镜像，`o` 是结果中心。

## 从未成功克隆的首次安装

若原 `project/AAOCA/start_aaoca_server.py` 不存在，不运行旧版过渡入口。先选择不存在的目标目录；保留失败克隆供排查，不覆盖已有目录：

```bash
cd /data5/zhougaowei/zhangruichen_workspace/aaoca_code
(
  set -eu
  export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
  export GIT_SSH_COMMAND="ssh -F /dev/null -i '$HOME/.ssh/project_readonly_ed25519' -p 443 -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o 'ProxyCommand=nc -X connect -x 127.0.0.1:17891 %h %p'"
  timeout 180 git clone --filter=blob:none --sparse --single-branch --branch main \
    ssh://git@ssh.github.com:443/tianwuasd/project.git project
  timeout 180 git -C project sparse-checkout set AAOCA
)
cd project/AAOCA
python3 setup_aaoca_environment.py
bash start_aaoca_server.sh --configure-network --github-proxy http://127.0.0.1:17891 --github-transport github-ssh --github-ssh-key "$HOME/.ssh/project_readonly_ed25519"
bash start_aaoca_server.sh
```

环境安装按独立 PyPI 镜像顺序进行；已存在的环境不会覆盖。只有源代码下载、配置、环境安装完成后才可新建预处理任务；本页不证明服务器已执行这些操作。
