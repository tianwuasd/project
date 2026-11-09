import hashlib
import json
from pathlib import Path

from chd_stunet.config import ROOT

from . import PROTOCOL


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=True,
                                     allow_nan=False).encode()).hexdigest()


def signed(record):
    return dict(record, identity=digest(record))


def check_signature(record):
    if record['identity'] != digest({k: v for k, v in record.items() if k != 'identity'}):
        raise ValueError('MedSAM2 任务签名摘要不一致')


def settings(smoke=False, epochs=None):
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(OmegaConf.load(ROOT/'configs/models/medsam2.yaml'), resolve=True)
    short = config.pop('smoke')
    if smoke:
        config.update(short)
    if epochs is not None:
        config['epochs'] = epochs
    config['scope'] = 'smoke' if smoke else 'formal'
    if config['protocol'] != PROTOCOL or config['epochs'] < 1:
        raise ValueError('MedSAM2 协议或训练轮数无效')
    if config['image_size'] % 32 or config['frames'] < 2 or not 1 <= config['max_objects'] <= 3:
        raise ValueError('MedSAM2 图像尺寸、帧数或目标数量无效')
    return config
