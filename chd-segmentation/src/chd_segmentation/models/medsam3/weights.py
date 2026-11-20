from pathlib import Path
import subprocess
import sys

from . import UPSTREAM_COMMIT


def file_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    if not path.is_file() or stat.st_size <= 0:
        raise ValueError('模型依赖文件为空或不是普通文件')
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def dependencies(config):
    root = Path(config['official_root']).resolve(strict=True)
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError('MedSAM3官方源码提交与固定版本不同')
    status = subprocess.check_output(
        ['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if status:
        raise ValueError('固定官方源码存在修改')
    return dict(upstream_commit=commit, files={key: file_identity(config[key]) for key in
        ('base_weights', 'lora_weights', 'official_config', 'bpe_path', 'python')})


def import_official(config):
    root = Path(config['official_root']).resolve(strict=True)
    sys.path.insert(0, str(root))
    import infer_sam
    import lora_layers
    import sam3.model_builder
    for module in (infer_sam, lora_layers, sam3.model_builder):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError('导入的官方模块不属于指定源码目录')
    return infer_sam, lora_layers, sam3.model_builder
