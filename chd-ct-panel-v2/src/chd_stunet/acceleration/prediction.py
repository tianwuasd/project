import os
from pathlib import Path
import subprocess
import time

import psutil

from .preprocess_cpu import GIB, admits, policy, snapshot


def budget(estimates, maximum, threads):
    return policy(snapshot(), [{'estimate': n} for n in estimates], maximum=maximum, case_threads=threads)


def can_admit(stage, fresh, reservations, estimate, threads):
    active = [row for row in reservations if row['threads'] > 0]
    idle = [row for row in reservations if row['threads'] == 0]
    # 驻留模型占内存，但空闲 worker 不占病例计算槽。
    stage = dict(stage, memory_budget_bytes=max(0, stage['memory_budget_bytes']-sum(max(r['estimate'], r['private']) for r in idle)))
    fresh = dict(fresh, memory_budget_bytes=max(0, fresh['memory_budget_bytes']-sum(max(0, r['estimate']-r['private']) for r in idle)))
    return admits(stage, fresh, active, estimate, threads)


def reservation(process, born, estimate, threads):
    private = 0
    if process.poll() is None:
        item = psutil.Process(process.pid)
        if item.create_time() == born:
            info = item.memory_full_info()
            private = int(getattr(info, 'uss', 0))
    return dict(estimate=estimate, threads=threads, private=private)


def select_visible(source, selection, maximum):
    from ..training.fleet import query_visible
    rows = query_visible(source)
    if selection != 'auto':
        ordinals = [int(v) for v in selection.split(',')]
        if len(set(ordinals)) != len(ordinals) or any(n < 0 for n in ordinals):
            raise ValueError('--gpus 须为 auto 或不重复的可见设备编号')
        if set(ordinals)-{r['ordinal'] for r in rows}:
            raise ValueError('指定 GPU 超出获准可见范围')
        rows = [r for r in rows if r['ordinal'] in ordinals]
    if not rows:
        raise ValueError('没有可见 CUDA 设备；CPU 短测须显式指定 --device cpu')
    return rows


def desktop_identity(pid, smoke):
    if os.name != 'nt' or not smoke:
        raise ValueError('桌面共享仅允许显式 Windows smoke 测试')
    process = psutil.Process(pid)
    executable = process.exe()
    if Path(executable).name.lower() != 'chatgpt.exe':
        raise ValueError('仅允许用户授权的 Codex 桌面进程')
    return dict(pid=pid, born=process.create_time(), executable=executable)


def permitted_desktop(identity):
    if not identity:
        return set()
    if os.name != 'nt':
        raise ValueError('服务器不接受桌面共享例外')
    for process in psutil.process_iter(['pid', 'create_time']):
        if process.info['pid'] == identity['pid']:
            if (process.info['create_time'] != identity['born'] or
                    os.path.normcase(process.exe()) != os.path.normcase(identity['executable'])):
                raise ValueError('桌面进程身份改变，拒绝共享')
            return {identity['pid']}
    return set()


def sharing_inventory(rows, identity):
    allowed = permitted_desktop(identity)
    return [dict(row, external_pids=[p for p in row['external_pids'] if p not in allowed],
                 shared_desktop_pids=sorted(set(row['external_pids']) & allowed)) for row in rows]


def admit_prediction_device(requested, shared=None):
    from ..training.resources import GPUUnavailable, admit_device
    if not shared:
        return admit_device(requested)
    import torch
    from ..training.fleet import query_live
    device = torch.device(requested)
    if device.type != 'cuda':
        raise ValueError('桌面共享只适用于 CUDA 测试')
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    uuid = str(properties.uuid)
    if not uuid.startswith('GPU-'):
        uuid = 'GPU-'+uuid
    row = sharing_inventory(query_live([{'uuid': uuid, 'ordinal': device.index}]), shared)[0]
    if set(row['external_pids'])-{os.getpid()} or row['usable_bytes'] <= 0:
        raise GPUUnavailable('测试GPU存在未授权占用或显存不足')
    free, total = torch.cuda.mem_get_info(device)
    reserve = max(4*GIB, int(total*.1))
    if free <= reserve:
        raise GPUUnavailable('测试GPU不满足显存保留要求')
    return device, dict(row, free_bytes=free, reserve_bytes=reserve, usable_bytes=free-reserve,
                        desktop_sharing=shared)


def process_options():
    if os.name == 'nt':
        return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {'start_new_session': True}


def stop_owned(entry):
    # 只使用实际 Popen 身份及出生时间，不按进程名或全局 GPU PID 清理。
    process = entry['process']
    if process.poll() is None:
        item = psutil.Process(process.pid)
        if item.create_time() != entry['born']:
            raise RuntimeError('进程出生身份变化，拒绝发送终止信号')
        item.terminate()
        done, alive = psutil.wait_procs([item], timeout=5)
        for child in alive:
            if child.create_time() == entry['born']:
                child.kill()
        process.wait(timeout=10)
    entry['stream'].close()


def resource_stop_reason(fresh, baseline):
    available = fresh.get('available_memory_bytes')
    if available is None:
        raise RuntimeError('缺少主机可用内存证据')
    reserve = max(4*GIB, int(baseline['available_memory_bytes']*.2))
    return 'host_memory_pressure' if available < reserve else None
