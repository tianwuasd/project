"""Pinned author weights, tensor-only loading and explicit target-head reset."""
import hashlib
from pathlib import Path

WEIGHT_SHA256 = 'f440f401bf4cac1d1d6f7c4635542ef057193a5f96c0b3c661a6179afeab7be6'
WEIGHT_BYTES = 117497661
WEIGHT_URL = 'https://huggingface.co/ziyanhuang/STU-Net/resolve/75d046881d0273899b65bf3ef1ee601e9973e7c8/small_ep4k.model'


def ensure_weights(path, env=None):
    """Worker-only download; authenticate before publishing, preserve existing bytes."""
    import os
    import tempfile
    import time
    import urllib.request
    from filelock import FileLock
    from runtime_setup import _install_routes, report_progress
    env = os.environ.copy() if env is None else dict(env)
    path = Path(path).expanduser().resolve()
    def valid(candidate):
        if candidate.stat().st_size != WEIGHT_BYTES:
            return False
        with candidate.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest() == WEIGHT_SHA256
    if path.exists():
        if not valid(path):
            raise ValueError('已有作者权重大小/SHA256不符；保留原件，请核查路径')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 900
    failures = []
    for route, proxy in _install_routes(env):
        temporary = None
        try:
            report_progress(env, 'pretrained_weights', '核验并准备STU-Net-S作者权重：' + route)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler(
                {'http': proxy, 'https': proxy} if proxy else {}))
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                with opener.open(WEIGHT_URL, timeout=20) as response:
                    received = 0
                    while True:
                        if time.monotonic() > deadline:
                            raise TimeoutError('权重下载超过900秒总预算')
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        received += len(block)
                        if received > WEIGHT_BYTES:
                            raise ValueError('下载大小超过固定作者权重')
                        stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if not valid(temporary):
                raise ValueError('下载权重SHA256/大小不符')
            with FileLock(str(path) + '.lock', timeout=20):
                if path.exists():
                    if not valid(path):
                        raise ValueError('并发创建的权重不符；保留原件')
                else:
                    os.replace(temporary, path)
            return path
        except (OSError, ValueError) as error:
            failures.append(route + ': ' + str(error))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError('作者权重未就绪，停止而不随机初始化：' + '; '.join(failures))


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ensure', required=True)
    ensure_weights(parser.parse_args().ensure)
