"""Pinned author weights, tensor-only loading and explicit target-head reset."""
import hashlib
from pathlib import Path

WEIGHT_SHA256 = 'f440f401bf4cac1d1d6f7c4635542ef057193a5f96c0b3c661a6179afeab7be6'
WEIGHT_BYTES = 117497661
WEIGHT_URL = 'https://huggingface.co/ziyanhuang/STU-Net/resolve/75d046881d0273899b65bf3ef1ee601e9973e7c8/small_ep4k.model'


def ensure_weights(path,proxy=None):
    """Explicit installation helper; stdlib-only, verified atomic cache write."""
    import os
    import urllib.request
    from uuid import uuid4
    from ..observability.locking import run_lock
    path=Path(path).expanduser().resolve();path.parent.mkdir(parents=True,exist_ok=True)
    with run_lock(path.with_suffix('.download.lock')):
        if path.is_file():
            with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
            if sha!=WEIGHT_SHA256 or path.stat().st_size!=WEIGHT_BYTES:
                raise ValueError('Existing weight cache differs; preserved, not overwritten')
            return dict(path=str(path),sha256=sha,reused=True)
        temporary=path.with_suffix('.'+uuid4().hex+'.partial')
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy})) if proxy else urllib.request.build_opener()
        try:
            digest=hashlib.sha256();size=0
            with opener.open(WEIGHT_URL,timeout=60) as response,temporary.open('wb') as stream:
                while chunk:=response.read(1024*1024):
                    stream.write(chunk);digest.update(chunk);size+=len(chunk)
                    if size>WEIGHT_BYTES:raise ValueError('Weight download exceeds pinned size')
            if size!=WEIGHT_BYTES or digest.hexdigest()!=WEIGHT_SHA256:raise ValueError('Downloaded weight SHA256/size differs')
            os.replace(temporary,path)
            return dict(path=str(path),sha256=WEIGHT_SHA256,reused=False)
        finally:
            if temporary.exists():temporary.unlink()


def load_pretrained(network, path):
    import numpy as np
    import torch
    path = Path(path).expanduser().resolve(strict=True)
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    if sha != WEIGHT_SHA256 or path.stat().st_size != WEIGHT_BYTES:
        raise ValueError('STU-Net-S作者权重SHA256/大小不符；停止，不退回随机初始化')
    # Legacy checkpoint has numpy scalar log entries. Permit only these known
    # numeric types after authenticating the complete file; no unrestricted pickle.
    core = getattr(np, '_core', np.core)
    allowed = [(core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.dtype,
               type(np.dtype(np.float32)), type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    source = checkpoint['state_dict']
    own = network.state_dict()
    expected = {k for k in own if not k.startswith('seg_outputs.')}
    actual = {k for k in source if not k.startswith('seg_outputs.')}
    if actual != expected or any(source[k].shape != own[k].shape for k in expected):
        raise ValueError('预训练骨干层名或形状不一致；停止迁移')
    missing, unexpected = network.load_state_dict({k: source[k] for k in expected}, strict=False)
    if unexpected or set(missing) != set(own) - expected:
        raise ValueError('预训练只允许重置目标输出头')
    if not all(torch.equal(network.state_dict()[k].cpu(), source[k]) for k in expected):
        raise ValueError('预训练加载后数值核验失败')
    return dict(sha256=sha, loaded_tensors=len(expected),
                loaded_elements=sum(source[k].numel() for k in expected),
                reset_heads=sorted(missing), all_backbone_tensors_exact=True)
