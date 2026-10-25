# 旧自动更新不可用时：直接Git引导升级

服务器历史记录确认的Git根目录如下；Vista子目录会在这次快进/稀疏检出后出现。以下不是本机Windows命令。执行前先结束该runtime活动任务，检查 `git status --short`，本地修改先自行保存；不使用reset/clean或强推。

```bash
cd /data5/zhougaowei/zhangruichen_workspace/chd_ct_code/project
git status --short
```

工作区干净后执行。使用服务器已有密钥和17891代理，不改变全局Git设置；任何一步失败会停止本次命令组。

```bash
(
  set -e
  export GIT_SSH_COMMAND="ssh -i \"$HOME/.ssh/project_readonly_ed25519\" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -p 443 -o 'ProxyCommand=nc -w 15 -X connect -x 127.0.0.1:17891 %h %p'"
  git fetch ssh://git@ssh.github.com:443/tianwuasd/project.git main
  git merge --ff-only FETCH_HEAD
  if [ "$(git config --bool core.sparseCheckout || true)" = true ]; then
    git sparse-checkout add vista3d_local_expert_refinement
  fi
  cd vista3d_local_expert_refinement
  bash start_server.sh --loaded-release
)
```

`--loaded-release` 只用于这次直接打开已拉取的新入口，避免旧runtime仍指向旧受管release。面板标题应为v0.3.0。进入 `c` 核对/保存原runtime网络，随后用 `u` 发布新受管版本，再按提示重开。之后使用普通 `bash start_server.sh`。若原来指定过 `--runtime`，以上及以后所有面板命令必须使用同一个原路径。

也可在新入口用以下参数保存网络并更新受管指针（先核对历史记录中的代理/密钥仍可用）：

```bash
cd /data5/zhougaowei/zhangruichen_workspace/chd_ct_code/project/vista3d_local_expert_refinement
bash start_server.sh --loaded-release --configure-network \
  --github-transport github-ssh \
  --github-proxy http://127.0.0.1:17891 \
  --github-ssh-key "$HOME/.ssh/project_readonly_ed25519"
bash start_server.sh --loaded-release --update
bash start_server.sh
```

这三条在 `project/vista3d_local_expert_refinement` 中执行。旧任务仍使用原冻结快照；从n新建的任务才使用新的GPU表单与内部执行流程。`w` 上传已提交代码，`v` 预览已完成任务产物；两者不是u的别名。
