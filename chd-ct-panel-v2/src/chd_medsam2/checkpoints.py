import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

from chd_stunet.observability.runlog import write_json

from . import PROTOCOL
from .contracts import check_signature, file_sha, read_json


class CheckpointMirrorError(OSError):
    pass


def coordinates(pointer):
    name = pointer['name']
    match = re.fullmatch(r'state-(\d+)-(\d+)-(validation|training)(?:-[a-f0-9]+)?\.pt', name)
    if match is None:
        raise ValueError('MedSAM2 检查点文件名无效')
    return int(match[1]), int(match[2]), match[3]


def progress(pointer):
    update, _, phase = coordinates(pointer)
    return update, phase == 'training'


def member(pointer):
    if (not isinstance(pointer, dict) or not isinstance(pointer.get('name'), str)
            or any(not isinstance(pointer.get(key), str) or not re.fullmatch(r'[a-f0-9]{64}', pointer[key])
                   for key in ('sha256', 'identity'))):
        raise ValueError('MedSAM2 检查点索引缺少有效的文件名或摘要')
    progress(pointer)
    return {key: pointer[key] for key in ('name', 'sha256', 'identity')}


def roles(root):
    index = read_json(root/'latest.json')
    saved = {'latest': member(index)}
    for role in ('previous', 'best'):
        if role in index:
            if index[role] is not None:
                saved[role] = member(index[role])
        elif (root/(role+'.json')).is_file():
            saved[role] = member(read_json(root/(role+'.json')))
    if any(pointer['identity'] != saved['latest']['identity'] for pointer in saved.values()):
        raise ValueError('MedSAM2 检查点索引混入不同训练任务')
    return saved


def publish_index(root, saved):
    # latest.json 原子提交整个索引，其余两个文件便于独立检查。
    write_json(root/'latest.json', dict(saved['latest'], previous=saved.get('previous'), best=saved.get('best')))
    for role in ('previous', 'best'):
        if role in saved:
            write_json(root/(role+'.json'), saved[role])


def shared_records(*indexes):
    records = {}
    for index in indexes:
        for pointer in index.values():
            current = records.setdefault(progress(pointer), pointer)
            if current['identity'] != pointer['identity'] or current['sha256'] != pointer['sha256']:
                raise ValueError('MedSAM2 同一训练位置存在不同的检查点 SHA256 或任务身份')
    return records


def matches(path, pointer):
    return path.is_file() and file_sha(path) == pointer['sha256']


def copy_archive(source, target, pointer):
    partial = target.with_suffix('.pt.partial')
    with source.open('rb') as reader, partial.open('wb') as writer:
        shutil.copyfileobj(reader, writer, length=8*1024**2)
        writer.flush()
        os.fsync(writer.fileno())
    if not matches(partial, pointer):
        raise ValueError('MedSAM2 检查点复制后 SHA256 不匹配')
    os.replace(partial, target)


def reconcile_recovery(local, output, contract):
    source = Path(local)/'checkpoints'
    mirror = Path(output)/'checkpoints'
    local_index = roles(source) if (source/'latest.json').is_file() else {}
    mirror_index = roles(mirror) if (mirror/'latest.json').is_file() else {}
    if not local_index and not mirror_index:
        return None
    if any(index['latest']['identity'] != contract['identity'] for index in (local_index, mirror_index) if index):
        raise ValueError('MedSAM2 恢复索引的训练任务身份不匹配')
    shared_records(local_index, mirror_index)
    remote_newer = bool(mirror_index and (
        not local_index or progress(mirror_index['latest']) > progress(local_index['latest'])))
    selected = mirror_index if remote_newer else local_index
    mirror_members = shared_records(mirror_index)
    repairs = {}
    # 先核对整个保留集合，再写入，禁止从已知损坏的最新版本静默退回。
    for record in {item['name']: item for item in selected.values()}.values():
        target = source/record['name']
        if not matches(target, record):
            peer = mirror_members.get(progress(record))
            if peer is None or not matches(mirror/peer['name'], record):
                raise ValueError('MedSAM2 没有可验证的检查点副本：'+record['name'])
            repairs[target] = (mirror/peer['name'], record)
    if repairs or remote_newer:
        source.mkdir(exist_ok=True)
        for target, (origin, record) in repairs.items():
            copy_archive(origin, target, record)
        publish_index(source, selected)
    return dict(source='mirror' if remote_newer else 'local', latest=selected['latest'],
                repaired=sorted(target.name for target in repairs))


def mirror_checkpoints(local, output):
    source = Path(local)/'checkpoints'
    target = Path(output)/'checkpoints'
    try:
        target.mkdir(exist_ok=True)
        saved = roles(source)
        if (target/'latest.json').is_file():
            remote = roles(target)
            local_latest, remote_latest = saved['latest'], remote['latest']
            if (local_latest['identity'] != remote_latest['identity']
                    or progress(remote_latest) > progress(local_latest)):
                raise ValueError('MedSAM2 结果目录已有其他任务或更新的检查点')
            shared_records(saved, remote)
        for record in saved.values():
            path = target/record['name']
            if not matches(path, record):
                copy_archive(source/record['name'], path, record)
        publish_index(target, saved)
        retained = {pointer['name'] for pointer in saved.values()}
        for path in target.glob('state-*.pt'):
            if path.name not in retained:
                path.unlink()
    except OSError as error:
        raise CheckpointMirrorError(error.errno, '检查点已在本地提交，结果目录镜像暂未同步') from error


def save(output, contract, network, optimizer, state, *, best=False, recovery=None):
    import torch

    root = Path(recovery or output)/'checkpoints'
    root.mkdir(exist_ok=True)
    name = f"state-{state['update']:08d}-{state['epoch']:04d}-{state['phase']}-{uuid4().hex[:12]}.pt"
    saved = roles(root) if (root/'latest.json').is_file() else {}
    latest = saved.get('latest')
    if latest and (latest['identity'] != contract['identity'] or progress(latest) > progress({'name': name})):
        raise ValueError('MedSAM2 检查点任务身份或更新顺序不一致')
    if latest and progress(latest) == progress({'name': name}):
        if file_sha(root/latest['name']) != latest['sha256']:
            raise ValueError('MedSAM2 当前检查点已损坏')
        if recovery:
            mirror_checkpoints(recovery, output)
        return latest
    path, temporary = root/name, root/(name+'.partial')
    with temporary.open('wb') as stream:
        torch.save(dict(protocol=PROTOCOL, contract=contract, state=state,
                        model=network.state_dict(), optimizer=optimizer.state_dict(),
                        torch_rng=torch.random.get_rng_state(),
                        device_rng=torch.cuda.get_rng_state(network.device) if network.device.type == 'cuda' else None),
                   stream)
        stream.flush()
        os.fsync(stream.fileno())
    pointer = dict(name=name, sha256=file_sha(temporary), identity=contract['identity'])
    os.replace(temporary, path)
    saved['latest'] = pointer
    if latest:
        saved['previous'] = latest
    if best:
        saved['best'] = pointer
    publish_index(root, saved)
    protected = {item['name'] for item in saved.values()}
    for item in root.glob('state-*.pt'):
        if item.name not in protected:
            item.unlink()
    if recovery:
        mirror_checkpoints(recovery, output)
    return pointer


def load(output, contract, role='latest'):
    return load_committed(output, contract, checkpoint_pointer(output, role))


def checkpoint_pointer(output, role):
    pointer = roles(Path(output)/'checkpoints').get(role)
    if pointer is None:
        raise FileNotFoundError(f'MedSAM2 检查点角色尚未提交：{role}')
    return pointer


def load_committed(output, contract, pointer):
    import torch

    root = Path(output)/'checkpoints'
    path = root/pointer['name']
    if (pointer['identity'] != contract['identity'] or path.parent != root
            or file_sha(path) != pointer['sha256']):
        raise ValueError('MedSAM2 检查点身份或 SHA256 校验失败')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    check_signature(payload['contract'])
    if payload['protocol'] != PROTOCOL or payload['contract'] != contract:
        raise ValueError('MedSAM2 检查点与当前模型、数据或配置不兼容')
    state = payload['state']
    update, epoch, phase = coordinates(pointer)
    config = contract['config']
    batch = config['global_batch']
    cases = len(contract['train_ids'])
    if ((state['update'], state['epoch'], state['phase']) != (update, epoch, phase)
            or state['sequence'] != update*batch
            or update > (config['epochs']*cases+batch-1)//batch
            or epoch != min(config['epochs'], state['sequence']//cases)):
        raise ValueError('MedSAM2 检查点进度与索引或训练合同不一致')
    return payload
