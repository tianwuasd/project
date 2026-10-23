"""One shared, measured resource budget and allocated GPU UUID intersection."""
import math
import os
import subprocess
from pathlib import Path

GIB = 2**30


def group_paths(root='/sys/fs/cgroup', membership='/proc/self/cgroup'):
    root = Path(root)
    paths = {root, root / 'memory', root / 'cpu', root / 'cpu,cpuacct'}
    try:
        for line in Path(membership).read_text().splitlines():
            _, controllers, relative = line.split(':', 2)
            relative = relative.lstrip('/')
            if '..' in Path(relative).parts:
                continue
            bases = [root] if not controllers else [root / controllers, *(root / c for c in controllers.split(','))]
            for base in bases:
                current = base / relative
                while current == base or current.is_relative_to(base):
                    paths.add(current)
                    if current == base:
                        break
                    current = current.parent
    except (OSError, ValueError):
        pass
    return sorted(paths)


def budget(settings, root='/sys/fs/cgroup', membership='/proc/self/cgroup'):
    import psutil
    process = psutil.Process()
    try:
        affinity = process.cpu_affinity()
    except (AttributeError, psutil.Error):
        affinity = list(range(os.cpu_count() or 1))
    cpu = float(len(affinity))
    memory = int(psutil.virtual_memory().available)
    for folder in group_paths(root, membership):
        try:
            quota, period = (folder / 'cpu.max').read_text().split()
            if quota != 'max' and int(quota) > 0 and int(period) > 0:
                cpu = min(cpu, int(quota) / int(period))
        except (OSError, ValueError):
            pass
        try:
            quota = int((folder / 'cpu.cfs_quota_us').read_text())
            period = int((folder / 'cpu.cfs_period_us').read_text())
            if quota > 0 and period > 0:
                cpu = min(cpu, quota / period)
        except (OSError, ValueError):
            pass
        for limit, used in [('memory.max', 'memory.current'), ('memory.limit_in_bytes', 'memory.usage_in_bytes')]:
            try:
                memory = min(memory, max(0, int((folder / limit).read_text()) - int((folder / used).read_text())))
            except (OSError, ValueError):
                pass
    load = psutil.cpu_percent(interval=.2, percpu=True)
    idle = sum(1 - load[i] / 100 for i in affinity if i < len(load))
    cores = max(1, math.floor(min(cpu, max(1, idle)) * .9))
    if settings.get('cpu_limit'):
        cores = min(cores, int(settings['cpu_limit']))
    ram = int(memory * .85)
    if settings.get('memory_gib'):
        ram = min(ram, int(settings['memory_gib'] * GIB))
    # A TotalSegmentator process can have resampling and save workers as well
    # as a main thread. Charge two thread groups to every active case.
    slots = min(settings.get('workers') or 8, max(1, cores // 2), settings.get('io_workers') or 4)
    threads = min(settings.get('threads') or 8, max(1, cores // (2 * slots)))
    return {'cpu_limit': cores, 'case_slots': slots, 'threads': threads,
            'memory_bytes': ram, 'io_slots': slots, 'cpu_quota': cpu,
            'note': 'shared estimates, not hard RSS isolation or an exclusive reservation'}


def estimate(geometry):
    shape, spacing = geometry['shape'], geometry['spacing_mm']
    native = math.prod(shape)
    highres = math.prod(max(1, math.ceil(n * s / .5)) for n, s in zip(shape, spacing))
    return int(2 * GIB + native * 32 + max(native, highres) * 40)


def select_devices(info, candidates):
    devices = {d['index']: d for d in info.get('devices', [])}
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError('请明确选择获准的可见 GPU 编号，不能重复或留空')
    selected = []
    for index in candidates:
        item = devices.get(index)
        if not item or not str(item.get('uuid', '')).startswith('GPU-'):
            raise ValueError('GPU 不可见或缺少完整 UUID；请重新检查环境（暂不支持 MIG）')
        selected.append({'index': index, 'uuid': item['uuid'], 'name': item.get('name')})
    return selected


def available_devices(selected, current, minimum_gib=8):
    visible = {d['uuid']: d for d in current.get('devices', [])}
    if any(d['uuid'] not in visible for d in selected):
        raise ValueError('继承的 GPU 可见范围/分配已变化；不能扩大范围续跑，请新建任务')
    try:
        query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free,utilization.gpu',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
        processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return []  # Unknown occupancy is not evidence of an idle allocated GPU.
    occupied = {line.split(',')[0].strip() for line in processes.stdout.splitlines() if ',' in line}
    free = set()
    for line in query.stdout.splitlines():
        try:
            uid, mb, utilization = [v.strip() for v in line.split(',')]
            if float(mb) >= minimum_gib * 1024 and float(utilization) <= 10 and uid not in occupied:
                free.add(uid)
        except ValueError:
            continue
    return [{'index': visible[d['uuid']]['index'], 'uuid': d['uuid']} for d in selected if d['uuid'] in free]


def child_env(threads, device=None):
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
        env[key] = str(threads)
    # Restrict to a verified UUID, never reset or expand inherited visibility.
    if device:
        env['CUDA_VISIBLE_DEVICES'] = device['uuid']
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTHONUNBUFFERED'] = '1'
    return env
