import os
from pathlib import Path
import shutil
from uuid import uuid4

import torch

from chd_stunet.observability.runlog import write_json

from . import PROTOCOL
from .contracts import check_signature, file_sha, read_json


class CheckpointMirrorError(OSError):
    pass


def mirror_checkpoints(local, output):
    source = Path(local)/'checkpoints'
    target = Path(output)/'checkpoints'
    try:
        target.mkdir(exist_ok=True)
        pointers = [name for name in ('latest.json', 'best.json') if (source/name).is_file()]
        for name in pointers:
            record = read_json(source/name)
            path = target/record['name']
            if not path.is_file() or file_sha(path) != record['sha256']:
                partial = path.with_suffix('.partial')
                with (source/record['name']).open('rb') as reader, partial.open('wb') as writer:
                    shutil.copyfileobj(reader, writer, length=8*1024**2)
                    writer.flush()
                    os.fsync(writer.fileno())
                if file_sha(partial) != record['sha256']:
                    raise ValueError('MedSAM2 检查点镜像摘要不匹配')
                os.replace(partial, path)
            write_json(target/name, record)
        retained = {p.name for p in source.glob('state-*.pt')}
        for path in target.glob('state-*.pt'):
            if path.name not in retained:
                path.unlink()
    except OSError as error:
        raise CheckpointMirrorError(error.errno, '检查点已在本地提交，结果目录镜像暂未同步') from error


def save(output, contract, network, optimizer, state, *, best=False, recovery=None):
    root = Path(recovery or output)/'checkpoints'
    root.mkdir(exist_ok=True)
    name = f"state-{state['update']:08d}-{state['epoch']:04d}-{state['phase']}-{uuid4().hex[:12]}.pt"
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
    write_json(root/'latest.json', pointer)
    if best:
        write_json(root/'best.json', pointer)
    protected = {pointer['name']}
    if (root/'best.json').is_file():
        protected.add(read_json(root/'best.json')['name'])
    generations = sorted(root.glob('state-*.pt'), key=lambda item: item.stat().st_mtime_ns, reverse=True)
    protected.update(item.name for item in generations[:2])
    for item in generations:
        if item.name not in protected:
            item.unlink()
    if recovery:
        mirror_checkpoints(recovery, output)
    return pointer


def load(output, contract, role='latest'):
    root = Path(output)/'checkpoints'
    pointer = read_json(root/(role+'.json'))
    path = root/pointer['name']
    if (pointer['identity'] != contract['identity'] or path.parent != root
            or file_sha(path) != pointer['sha256']):
        raise ValueError('MedSAM2 检查点身份或 SHA256 校验失败')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    check_signature(payload['contract'])
    if payload['protocol'] != PROTOCOL or payload['contract'] != contract:
        raise ValueError('MedSAM2 检查点与当前模型、数据或配置不兼容')
    return payload
