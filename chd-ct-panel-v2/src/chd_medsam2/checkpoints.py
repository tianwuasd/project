import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import torch

from chd_stunet.observability.runlog import write_json

from . import PROTOCOL
from .contracts import check_signature, file_sha, read_json


class CheckpointMirrorError(OSError):
    pass


def progress(pointer):
    name = pointer['name']
    match = re.fullmatch(r'state-(\d+)-(\d+)-(validation|training)(?:-[a-f0-9]+)?\.pt', name)
    if match is None:
        raise ValueError('MedSAM2 检查点文件名无效')
    return int(match[1]), match[3] == 'training'


def member(pointer):
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
            local_members = {progress(pointer): pointer for pointer in saved.values()}
            for pointer in remote.values():
                current = local_members.get(progress(pointer))
                if current is not None and current['sha256'] != pointer['sha256']:
                    raise ValueError('MedSAM2 同一训练位置存在不同的检查点 SHA256')
        for record in saved.values():
            path = target/record['name']
            if not path.is_file() or file_sha(path) != record['sha256']:
                partial = path.with_suffix('.pt.partial')
                with (source/record['name']).open('rb') as reader, partial.open('wb') as writer:
                    shutil.copyfileobj(reader, writer, length=8*1024**2)
                    writer.flush()
                    os.fsync(writer.fileno())
                if file_sha(partial) != record['sha256']:
                    raise ValueError('MedSAM2 检查点镜像摘要不匹配')
                os.replace(partial, path)
        publish_index(target, saved)
        retained = {pointer['name'] for pointer in saved.values()}
        for path in target.glob('state-*.pt'):
            if path.name not in retained:
                path.unlink()
    except OSError as error:
        raise CheckpointMirrorError(error.errno, '检查点已在本地提交，结果目录镜像暂未同步') from error


def save(output, contract, network, optimizer, state, *, best=False, recovery=None):
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
    root = Path(output)/'checkpoints'
    pointer = roles(root).get(role)
    if pointer is None:
        raise FileNotFoundError(f'MedSAM2 检查点角色尚未提交：{role}')
    path = root/pointer['name']
    if (pointer['identity'] != contract['identity'] or path.parent != root
            or file_sha(path) != pointer['sha256']):
        raise ValueError('MedSAM2 检查点身份或 SHA256 校验失败')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    check_signature(payload['contract'])
    if payload['protocol'] != PROTOCOL or payload['contract'] != contract:
        raise ValueError('MedSAM2 检查点与当前模型、数据或配置不兼容')
    return payload
