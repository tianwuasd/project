"""V2 source maintenance belongs to the complete original V2 project/branch."""
from pathlib import Path
import sys
from contextlib import contextmanager

BRANCH='codex/ct-panel-v2'
SUBDIR='chd-ct-panel-v2'


def _host():
    root=Path(__file__).resolve().parents[2]
    source=root/'src'
    if root.name!=SUBDIR or not (source/'chd_ct_v2/panel_v2/identity.py').is_file():
        raise ValueError('更新/代码上传请从原CHD V2的start_server_v2.sh打开；冻结运行副本只用于计算')
    if str(source) not in sys.path:sys.path.insert(0,str(source))
    return root


def selected_entry(runtime):
    from v2_host_identity import check
    check(runtime)
    if (Path(runtime)/'release.json').exists():
        raise ValueError('发现旧独立引擎发布指针；整合版V2必须使用自己的新runtime')
    return None


@contextmanager
def task_guard(runtime):
    from task_runtime import runtime_lock,active_processes
    from pipeline_catalog import history
    with runtime_lock(Path(runtime)/'pipeline_v2'):
        if any(active_processes(row['directory']) for row in history(runtime)):
            raise ValueError('STU-Net仍有活动任务；停止并确认退出后再维护代码')
        yield


def update(runtime,results=None):
    from v2_host_identity import claim
    claim(runtime,results)
    root=_host()
    from chd_ct_v2.server import network,updater
    with task_guard(runtime),network.use(network.load(runtime)):
        repo=updater.inspect_repository(root)
        return updater.update(repo,Path(runtime))


def push_code(args,profile,interactive):
    _host()
    from v2_host_identity import claim
    claim(args.runtime or profile['runtime'],getattr(args,'results',None) or profile['results'])
    from chd_ct_v2.server import code_sync
    with task_guard(args.runtime or profile['runtime']):
        return code_sync.run(args,profile,interactive)
