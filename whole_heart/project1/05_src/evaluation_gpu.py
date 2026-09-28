"""只读探测GPU：多次采样、尊重可见范围，不结束外部进程。

这不是集群资源预订；管理员的分配规则仍然优先。
"""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time

CONFIG = Path(__file__).resolve().parents[1] / '06_configs/evaluation_gpu.json'


def load_settings():
    settings = json.loads(CONFIG.read_text(encoding='utf-8'))
    bounds = {'max_gpus': (1, 3), 'min_free_mib': (1024, 1048576),
              'max_utilization_percent': (0, 10), 'idle_samples': (2, 10),
              'sample_interval_seconds': (1, 30), 'retry_seconds': (1, 60)}
    for key, (low, high) in bounds.items():
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'评价GPU配置 {key} 应在 {low} 到 {high} 之间')
    for key in ['max_gpus', 'idle_samples']:
        if not isinstance(settings[key], int):
            raise ValueError(f'{key} 必须为整数')
    allowed = settings['allowed_gpus']
    if allowed is not None and (not isinstance(allowed, list) or not all(isinstance(x, str) for x in allowed)):
        raise ValueError('allowed_gpus 应为编号/完整UUID字符串列表，或 null')
    return settings


def query(fields, kind='gpu'):
    output = subprocess.check_output(['nvidia-smi', f'--query-{kind}={fields}',
                                     '--format=csv,noheader,nounits'], text=True, timeout=15)
    return [[x.strip() for x in row] for row in csv.reader(output.splitlines()) if row]


def snapshot():
    """字段不可读取就报错，不能把 N/A 或查询失败当作空闲。"""
    rows = query('index,uuid,memory.free,utilization.gpu')
    busy = {row[0] for row in query('gpu_uuid,pid', 'compute-apps')}
    if not rows or any(not uuid.startswith('GPU-') for uuid in busy):
        raise RuntimeError('无法可靠读取GPU/计算进程列表')
    result = []
    for index, uuid, free, utilization in rows:
        free, utilization = float(free), float(utilization)
        if not math.isfinite(free) or not math.isfinite(utilization):
            raise ValueError('GPU监测数值无效')
        result.append(dict(index=index, uuid=uuid, free_mib=free,
                           utilization=utilization, busy=uuid in busy))
    return result


def allowed_ids(rows, settings, environ=None):
    """取配置与外部可见范围的交集。空字符串/-1 表示没有卡，未知编号拒绝。"""
    env = os.environ if environ is None else environ
    allowed = {r['uuid'] for r in rows}
    restrictions = []
    if settings['allowed_gpus'] is not None:
        restrictions.append(settings['allowed_gpus'])
    # 入口自行缩小 CUDA_VISIBLE_DEVICES 前保存原值，防止误把训练第一张卡当成整个分配范围。
    cuda = env.get('WHOLE_HEART_EVAL_VISIBLE_DEVICES', env.get('CUDA_VISIBLE_DEVICES'))
    if cuda is not None and cuda != '__UNRESTRICTED__':
        restrictions.append(cuda.split(','))
    for name in ['NVIDIA_VISIBLE_DEVICES', 'SLURM_STEP_GPUS', 'SLURM_JOB_GPUS']:
        value = env.get(name)
        if value is not None and value != 'all':
            restrictions.append(value.split(','))
    for tokens in restrictions:
        selected = set()
        for token in tokens:
            token = token.strip()
            if token in {'', '-1', 'none', 'void'}:
                continue
            found = [r['uuid'] for r in rows if token in (r['index'], r['uuid'])]
            if len(found) != 1:
                raise ValueError(f'无法映射分配的GPU {token}；请明确填写 nvidia-smi 编号或完整UUID')
            selected.update(found)
        allowed.intersection_update(selected)
    return allowed


def idle_reason(row, settings):
    if row['busy']:
        return '已有计算进程（即使利用率为0也跳过）'
    if row['utilization'] > settings['max_utilization_percent']:
        return f"利用率 {row['utilization']:g}% 过高"
    if row['free_mib'] < settings['min_free_mib']:
        return f"空闲显存 {row['free_mib']:g} MiB 不足"
    return None


def idle_gpus(settings, candidates=None):
    """全部采样都通过才返回；返回顺序按最后一次空闲显存从高到低排列。"""
    eligible = None
    for sample in range(settings['idle_samples']):
        rows = snapshot()
        allowed = allowed_ids(rows, settings)
        if not allowed:
            raise RuntimeError('当前分配/配置范围没有GPU，请检查可见范围和 allowed_gpus')
        if candidates is not None:
            allowed.intersection_update(candidates)
        passing = set()
        for row in rows:
            if row['uuid'] not in allowed:
                continue
            reason = idle_reason(row, settings)
            if reason:
                print(f"GPU {row['index']} 跳过：{reason}", flush=True)
            else:
                passing.add(row['uuid'])
        eligible = passing if eligible is None else eligible & passing
        if sample + 1 < settings['idle_samples']:
            time.sleep(settings['sample_interval_seconds'])
    return [r['uuid'] for r in sorted(rows, key=lambda r: (-r['free_mib'], int(r['index'])))
            if r['uuid'] in eligible]
