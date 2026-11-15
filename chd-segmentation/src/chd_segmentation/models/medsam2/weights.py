from pathlib import Path

from ...artifacts.store import file_digest


OFFICIAL_WEIGHT = {
    'filename':'MedSAM2_latest.pt',
    'repository':'wanglab/MedSAM2',
    'revision':'e4a6f35edd7e091619cbc0750f462f1574e23955',
    'bytes':156040129,
    'sha256':'c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060',
    'url':'https://huggingface.co/wanglab/MedSAM2/resolve/e4a6f35edd7e091619cbc0750f462f1574e23955/MedSAM2_latest.pt',
}


def verify_official_weight(path):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size!=OFFICIAL_WEIGHT['bytes'] or file_digest(path)!=OFFICIAL_WEIGHT['sha256']:
        raise ValueError('MedSAM2官方初始化权重的容量或摘要不匹配')
    return dict(OFFICIAL_WEIGHT,path=str(path))
