"""Adapter to Vista's isolated releases; publication always uses the Git checkout."""
from pathlib import Path
import task_update
from server_data import read_json
from .launcher import ROOT

ORIGIN_URLS=task_update.ORIGINS


def project_checkout(runtime):
    pointer=Path(runtime)/'release.json'
    if pointer.exists():
        return Path(read_json(pointer)['repository'])/task_update.SUBDIR
    return ROOT


def inspect_repository(project=ROOT):
    git=task_update.git
    root=Path(git(project,'rev-parse','--show-toplevel').stdout).resolve()
    subtree=Path(project).resolve().relative_to(root).as_posix()
    if subtree != task_update.SUBDIR:
        raise ValueError('请从 project/vista3d_local_expert_refinement 的 Git 检出上传代码')
    origin=git(root,'config','--get','remote.origin.url').stdout.rstrip('/')
    if origin.removesuffix('.git') not in {x.rstrip('/').removesuffix('.git') for x in ORIGIN_URLS}:
        raise ValueError('origin不是已核对的tianwuasd/project仓库')
    if git(root,'branch','--show-current').stdout != 'main':
        raise ValueError('代码上传仅接受main分支，不自动切换')
    if git(root,'status','--porcelain','--untracked-files=normal').stdout:
        raise ValueError('仓库有未提交修改；请先整理并提交本项目代码，面板不会自动add/commit或覆盖')
    for marker in ['MERGE_HEAD','rebase-merge','rebase-apply','CHERRY_PICK_HEAD']:
        p=Path(git(root,'rev-parse','--git-path',marker).stdout)
        if (p if p.is_absolute() else root/p).exists():
            raise ValueError('请先处理未完成的Git操作')
    return dict(root=str(root),subtree=subtree,head=git(root,'rev-parse','HEAD').stdout)


def ancestor(root,old,new):
    return task_update.git(root,'merge-base','--is-ancestor',old,new,check=False).returncode == 0
