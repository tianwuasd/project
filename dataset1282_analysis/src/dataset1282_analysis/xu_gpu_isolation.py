import copy
import argparse
import json
import os
import re


def validate_entry(argv, environment, default='cuda'):
    if '--help' in argv or '-h' in argv:
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--device',default=default)
    args,_ = parser.parse_known_args(argv)
    if 'XU_GPU_ISOLATION' in environment:
        mapping = json.loads(environment['XU_GPU_ISOLATION'])
        mask = ','.join(r['uuid'] for r in mapping) if mapping else '-1'
        if environment.get('CUDA_VISIBLE_DEVICES') != mask:
            raise ValueError('计算入口的 GPU 隔离环境不匹配')
        if args.device not in [r['device'] for r in mapping] and args.device != 'cpu':
            raise ValueError('计算入口请求了未分配设备')
        return
    if args.device != 'cpu':
        raise ValueError('GPU 计算需要隔离启动，请使用 xu_pipeline.py 或 xu_fleet_probe.py')
    environment['CUDA_VISIBLE_DEVICES'] = '-1'


def authorized_devices(environment, inventory):
    physical = sorted(inventory.values(),key=lambda row:row['physical_index'])
    def resolve(mask, pool, numeric=False):
        if mask is None or mask.lower() == 'all':
            return list(pool)
        if mask.lower() in ('','-1','none','void'):
            return []
        result = []
        for token in mask.split(','):
            token = token.strip()
            if token.isdigit() and not numeric:
                raise ValueError('无法证明 CUDA 数字序号与物理卡的对应关系；授权范围请使用 GPU UUID')
            matches = [r for r in pool if (token.isdigit() and r['physical_index'] == int(token))
                       or (token.startswith('GPU-') and r['uuid'].startswith(token))]
            if len(matches) != 1 or matches[0] in result:
                raise ValueError('GPU 授权范围含未知、重复或不明确的设备: '+token)
            result.append(matches[0])
        return result
    container = resolve(environment.get('NVIDIA_VISIBLE_DEVICES'),physical,numeric=True)
    selected = resolve(environment.get('CUDA_VISIBLE_DEVICES'),container)
    return [dict(device=f'cuda:{i}',uuid=row['uuid'],name=row['name'],total_bytes=row['total_bytes'])
            for i,row in enumerate(selected)]


def coordinator_environment(environment, inventory):
    env = dict(environment)
    env['XU_COORDINATOR_DEVICES'] = json.dumps(authorized_devices(environment,inventory))
    env['XU_COORDINATOR_AUTH_MASK'] = environment.get('CUDA_VISIBLE_DEVICES','')
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    return env


def configure_coordinator(cpu_only=False):
    if 'XU_COORDINATOR_DEVICES' in os.environ:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
            raise ValueError('管理进程必须隐藏全部 CUDA 设备')
        return
    from .xu_gpu_probe import gpu_snapshot
    inventory = {} if cpu_only else gpu_snapshot()
    original = dict(os.environ)
    if cpu_only:
        original.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none')
    env = coordinator_environment(original,inventory)
    for name in ('XU_COORDINATOR_DEVICES','XU_COORDINATOR_AUTH_MASK','CUDA_VISIBLE_DEVICES'):
        os.environ[name] = env[name]


def launch_scope(argv, overlay, devices, environment):
    names = list(devices)
    env = dict(environment)
    for name in ('XU_COORDINATOR_DEVICES','XU_COORDINATOR_AUTH_MASK','XU_CUDA_ADMISSION',
                 'XU_CUDA_ADMISSIONS','XU_BRANCH_DEVICES','XU_REGISTERED_PIDS','XU_GPU_ISOLATION'):
        env.pop(name,None)
    env.update(overlay)
    mapping = []
    if names != ['cpu']:
        if not names or len(set(names)) != len(names) or not all(re.fullmatch(r'cuda:\d+',n) for n in names):
            raise ValueError('GPU 分配必须是明确且唯一的设备组')
        primary = json.loads(overlay.get('XU_CUDA_ADMISSION','null'))
        tickets = json.loads(overlay.get('XU_CUDA_ADMISSIONS','null')) or [primary]
        if primary is None or [t['device'] for t in tickets] != names or primary != tickets[0]:
            raise ValueError('隔离范围与 GPU 准入凭据不一致')
        uuids = [t['uuid'] for t in tickets]
        if len(set(uuids)) != len(uuids) or not all(u.startswith('GPU-') for u in uuids):
            raise ValueError('GPU 隔离需要唯一的物理 UUID')
        if 'XU_COORDINATOR_DEVICES' in environment:
            pool = {row['device']:row['uuid'] for row in json.loads(environment['XU_COORDINATOR_DEVICES'])}
            if any(pool.get(name) != uuid for name,uuid in zip(names,uuids)):
                raise ValueError('子进程分配超出管理进程的授权范围')
        mapping = [dict(parent_device=name,device=f'cuda:{i}',uuid=uuid)
                   for i,(name,uuid) in enumerate(zip(names,uuids))]
        mapped = [dict(ticket,device=row['device']) for ticket,row in zip(tickets,mapping)]
        env['XU_CUDA_ADMISSION'] = json.dumps(mapped[0])
        if len(mapped)>1:
            env['XU_CUDA_ADMISSIONS'] = json.dumps(mapped)
            env['XU_BRANCH_DEVICES'] = json.dumps([row['device'] for row in mapped])
    elif any(overlay.get(k) for k in ('XU_CUDA_ADMISSION','XU_CUDA_ADMISSIONS','XU_BRANCH_DEVICES')):
        raise ValueError('CPU 工作进程不能携带 GPU 准入凭据')
    env['CUDA_VISIBLE_DEVICES'] = ','.join(row['uuid'] for row in mapping) if mapping else '-1'
    env['XU_GPU_ISOLATION'] = json.dumps(mapping)
    translated = {row['parent_device']:row['device'] for row in mapping}
    args = []
    for value in map(str,argv):
        if value.startswith('cuda:') and value not in translated:
            raise ValueError('启动参数包含本组之外的 GPU: '+value)
        args.append(translated.get(value,value))
    return args,env,mapping


def admit_worker_scope(tickets, owned, inventory):
    from .xu_resources import _check_device, ResourceUnavailable
    for ticket in tickets:
        if ticket['uuid'] not in inventory:
            raise ResourceUnavailable('无法核实获分配 GPU 的物理身份')
        info = dict(inventory[ticket['uuid']],device=ticket['device'])
        _check_device(info,ticket['policy'],owned,ticket['require_pid_evidence'])


def verify_worker_scope():
    mapping = json.loads(os.environ['XU_GPU_ISOLATION'])
    expected = [r['uuid'] for r in mapping]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != (','.join(expected) if expected else '-1'):
        raise ValueError('工作进程的 CUDA 可见范围被修改')
    actual = []
    if expected:
        from .xu_gpu_probe import actual_visible_devices, gpu_snapshot
        from .xu_resources import registered_identities, _identities
        primary = json.loads(os.environ['XU_CUDA_ADMISSION'])
        tickets = json.loads(os.environ.get('XU_CUDA_ADMISSIONS','null')) or [primary]
        if [(t['device'],t['uuid']) for t in tickets] != [(r['device'],r['uuid']) for r in mapping]:
            raise ValueError('工作进程的隔离映射与准入凭据不匹配')
        members = json.loads(os.environ['XU_REGISTERED_PIDS'])
        owned = {**primary['owned'],**registered_identities(primary['owned'],members),**_identities([os.getpid()])}
        admit_worker_scope(tickets,owned,gpu_snapshot())
        actual = actual_visible_devices()
        if [row['uuid'] for row in actual] != expected:
            raise ValueError('实际 CUDA UUID 或顺序与隔离分配不一致')
    return dict(pid=os.getpid(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],mapping=mapping,
                expected_uuids=expected,actual_uuids=[r['uuid'] for r in actual],
                cuda_enumerated=bool(expected),state='verified')


def report_for_parent(report, devices):
    if list(devices) == ['cpu']:
        return copy.deepcopy(report)
    mapping = {f'cuda:{i}':name for i,name in enumerate(devices)}
    def convert(value):
        if isinstance(value,dict):
            return {mapping.get(k,k):convert(v) for k,v in value.items()}
        if isinstance(value,list):
            return [convert(v) for v in value]
        if isinstance(value,str):
            return mapping.get(value,value)
        return value
    return convert(report)
