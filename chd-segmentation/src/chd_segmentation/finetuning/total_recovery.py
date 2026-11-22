from time import monotonic

from ..observability.events import event


def memory_snapshot(session):
    with session.condition:
        state = dict(session.state)
    fields = ('sampled_at', 'sampled_monotonic', 'paused', 'pressure_reasons', 'monitor_failed',
              'monitor_error', 'capacity', 'task_rss', 'peak_task_rss', 'project_bytes',
              'limited_project_bytes', 'project_limit_bytes', 'system_headroom_bytes', 'pending_bytes',
              'memory_scopes')
    snapshot = {key: state.get(key) for key in fields}
    host = state.get('host', {})
    snapshot['available_bytes'] = host.get('available_bytes')
    snapshot['total_bytes'] = host.get('total_bytes')
    return snapshot


def check_monitor(session, run, stage):
    snapshot = memory_snapshot(session)
    if session.cancel.is_set():
        raise InterruptedError('Total任务已取消')
    if snapshot['monitor_failed']:
        event(run, 'total_resource_monitor_failed', stage=stage, memory=snapshot)
        if session.future.done():
            session.future.result()
        raise RuntimeError(f"Total资源监测异常：{snapshot['monitor_error']}")
    if monotonic()-snapshot['sampled_monotonic'] > 30:
        event(run, 'total_resource_monitor_stale', stage=stage, memory=snapshot)
        raise TimeoutError('Total资源监测超过30秒没有新样本')
    return snapshot


def prediction_ready(session, run):
    # 连续三个新样本均有余量，才开始重新加载模型和病例。
    healthy, previous = 0, None
    while healthy < 3:
        snapshot = check_monitor(session, run, 'predict')
        if snapshot['paused']:
            event(run, 'total_prediction_pressure', phase='before_start', memory=snapshot)
            return False
        if snapshot['sampled_monotonic'] != previous:
            healthy += 1
            previous = snapshot['sampled_monotonic']
        if healthy < 3 and session.cancel.wait(.2):
            raise InterruptedError('Total预测恢复等待已取消')
    event(run, 'total_prediction_ready', stable_samples=healthy, memory=snapshot)
    return True


def supervise_child(config, session, stage, process, members):
    began = monotonic()
    while process.poll() is None:
        for member in members(process.pid):
            pid = member['pid']
            if session.handle.members.get(pid) != member:
                session.handle.members[pid] = member
        snapshot = check_monitor(session, config['run_dir'], stage)
        if snapshot['paused']:
            event(config['run_dir'], 'total_resource_pressure', stage=stage,
                  child_pid=process.pid, seconds=monotonic()-began, memory=snapshot)
            if stage == 'predict':
                return False
            raise MemoryError(f'Total {stage}资源不足：{snapshot}')
        if session.cancel.wait(.5):
            raise InterruptedError('Total子进程监督已取消')
    if process.returncode:
        from subprocess import CalledProcessError
        raise CalledProcessError(process.returncode, process.args)
    return True
