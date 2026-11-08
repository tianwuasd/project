import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

from ..config import ROOT
from ..preprocessing.cohort import TARGETS
from ..training.checkpoints import read_checkpoint
from ..training.code_snapshot import freeze_code
from ..training.contracts import digest, scientific_sources


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inference_plan(plan, patch):
    # 原预处理计划保持只读；预测导出增加官方概率插值参数。
    result = copy.deepcopy(plan)
    config = result['configurations']['3d_fullres']
    config['patch_size'] = list(patch)
    config['resampling_fn_probabilities'] = 'resample_data_or_seg_to_shape'
    config['resampling_fn_probabilities_kwargs'] = {
        'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None}
    return result


def validate_producer(sources):
    audited = read_json(ROOT/'configs/prediction/compatibility.json')['sources']
    if digest(sources) not in audited:
        raise ValueError('来源训练代码尚未审计预测兼容；原模型保留')


def inspect_training(run, checkpoint='best', *, smoke=False, allow_intermediate=False):
    run = Path(run).resolve(strict=True)
    if checkpoint not in ('best', 'epoch', 'latest'):
        raise ValueError('检查点必须选择 best、epoch 或 latest')
    signature = read_json(run/'signature.json')
    if digest({k: v for k, v in signature.items() if k != 'identity'}) != signature['identity']:
        raise ValueError('训练签名摘要不一致')
    recipe = signature['recipe']
    if recipe['protocol'] != 'imagechd-stunet-s-ft-partial-v1':
        raise ValueError('训练协议不属于已审计七结构方法')
    if recipe['model'] != 'stunet-s' or recipe['nnunet_version'] != '2.8.1':
        raise ValueError('仅支持已审计的 STU-Net-S / nnU-Net 2.8.1')
    if importlib.metadata.version('nnunetv2') != recipe['nnunet_version']:
        raise ValueError('预测 nnU-Net 版本与模型要求不一致')
    for name in ('torch', 'nnunetv2', 'numpy', 'scipy', 'blosc2', 'acvl-utils'):
        if importlib.metadata.version(name) != signature['versions'][name]:
            raise ValueError('预测计算依赖与来源训练不同，需要兼容审计：'+name)
    if recipe['scope'] != 'formal' and not smoke:
        raise ValueError('短测模型只能用于 --smoke 预测')
    summary = read_json(run/'summary.json')
    if summary['status'] != 'completed' and not allow_intermediate:
        raise ValueError('训练尚未完成；阶段性预测需要 --allow-intermediate')
    if summary['status'] not in ('completed', 'stopped'):
        raise ValueError('来源训练须已完成或安全停止')
    freeze_code(run, signature, create=False)
    validate_producer(signature['producer_ast_sha256'])
    current = scientific_sources(ROOT)
    # 调度源码可以升级，但解释权重的网络源码必须逐项匹配。
    critical = ('src/chd_stunet/training/model.py', 'src/chd_stunet/_vendor/stunet_architecture.py')
    for name in critical:
        if current[name] != signature['producer_ast_sha256'].get(name):
            raise ValueError('网络解释代码尚未审计兼容：'+name)
    pointer = read_json(run/'checkpoints'/f'{checkpoint}.json')
    payload = read_checkpoint(run/'checkpoints', signature, f'{checkpoint}.json')
    data = signature['data']
    patch = recipe['patch_size']
    if len(patch) != 3 or any(type(n) is not int or n <= 0 for n in patch):
        raise ValueError('训练 patch 无效')
    from ..training.model import topology
    architecture = topology(data['plan']['configurations']['3d_fullres']['spacing'], patch)
    if architecture != recipe['architecture']:
        raise ValueError('检查点网络结构与原 spacing/patch 不匹配')
    record = dict(run=str(run), fold=data['fold'], checkpoint=checkpoint, pointer=pointer,
                  signature=signature, patch_size=patch, architecture=architecture,
                  plan=inference_plan(data['plan'], patch),
                  dataset_json={'channel_names': {'0': 'CT'}, 'labels': TARGETS,
                                'file_ending': '.nii.gz'},
                  training_status=summary['status'], epoch=payload['state']['epoch'],
                  source_scope=recipe['scope'])
    del payload
    return record


def freeze_weights(record, destination):
    import os
    import torch
    from ..observability.runlog import write_json
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    run = Path(record['run'])
    # 固定不可变文件，不再跟随可能被其他任务更新的 best 指针。
    path = run/'checkpoints'/record['pointer']['file']
    if file_sha(path) != record['pointer']['sha256']:
        raise ValueError('来源权重在冻结前发生改变')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['signature'] != record['signature']:
        raise ValueError('来源权重身份不一致')
    target = destination/'model.pt'
    temporary = destination/'model.partial'
    with temporary.open('wb') as stream:
        torch.save(payload['model'], stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    result = dict(record, weights=str(target), weights_sha256=file_sha(target))
    write_json(destination/'model.json', result)
    return result


def restore_network(record, device):
    import torch
    from ..training.model import build_model
    if file_sha(record['weights']) != record['weights_sha256']:
        raise ValueError('预测冻结权重校验失败')
    network = build_model(record['architecture'])
    network.load_state_dict(torch.load(record['weights'], map_location='cpu', weights_only=True), strict=True)
    network.decoder.deep_supervision = False
    network.eval().requires_grad_(False)
    return network.to(device)
