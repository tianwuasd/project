import argparse
from importlib import metadata
from importlib import util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

from filelock import FileLock
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from .assets import MED_COMMIT, MED_SHA, MED_URL, clone_official, download_file
from .install import install, validate_asset_destination
from .network import subprocess_environment


def dependency_status(code, target):
    requirements = [Requirement(line) for line in (code/'requirements-medsam2.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
    available = {canonicalize_name(dist.metadata['Name']): dist.version for dist in metadata.distributions()}
    if target.is_dir():
        available.update({canonicalize_name(dist.metadata['Name']): dist.version for dist in metadata.distributions(path=[str(target)])})
    missing = [str(req) for req in requirements if canonicalize_name(req.name) not in available or
               Version(available[canonicalize_name(req.name)]) not in req.specifier]
    torch = metadata.version('torch')
    return dict(torch=torch, torch_compatible=Version(torch) >= Version('2.11'), missing=missing, target=str(target))


def prepare_dependencies(code, root, target, proxy, allowed):
    state = dependency_status(code, target)
    if not state['torch_compatible']:
        raise ValueError('MedSAM2固定TensorDict需要PyTorch>=2.11，保留当前环境：'+state['torch'])
    if state['missing']:
        if not allowed:
            raise ValueError('缺少MedSAM2依赖；安装命令需明确指定--install-dependencies：'+json.dumps(state, ensure_ascii=False))
        if target.exists() and any(target.iterdir()):
            raise FileExistsError('既有额外依赖不完整或版本不同，保留原目录：'+str(target))
        temp = root/'course/chd-segmentation/installation-work'/uuid4().hex
        temp.mkdir(parents=True)
        target.mkdir(parents=True, exist_ok=True)
        env = dict(subprocess_environment(proxy), TMPDIR=str(temp), TEMP=str(temp), TMP=str(temp),
                   PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-B', '-m', 'pip', 'install', '--no-deps', '--no-cache-dir',
                   '--index-url', 'https://pypi.org/simple', '--target', str(target),
                   '-r', str(code/'requirements-medsam2.txt')]
        if proxy:
            command.extend(['--proxy', proxy])
        try:
            subprocess.run(command, env=env, check=True)
        finally:
            if temp.exists():
                if temp.is_symlink() or any(p.is_symlink() for p in temp.rglob('*')):
                    raise ValueError('依赖安装临时目录含链接，保留待核对')
                shutil.rmtree(temp)
        if dependency_status(code, target)['missing']:
            raise ValueError('额外依赖安装核验未通过')
    else:
        target.mkdir(parents=True, exist_ok=True)
    return dependency_status(code, target)


def existing_or_default(explicit, candidates, default):
    if explicit:
        return Path(explicit).resolve()
    found = [path for path in candidates if path.exists()]
    return found[0] if found else default


def main():
    parser = argparse.ArgumentParser(description='Q与MedSAM2既有材料核对、自动准备和配置生成')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('check', 'install'):
        child = commands.add_parser(name)
        child.add_argument('--root', default='/data4/zhangruichen')
        child.add_argument('--code', default=str(Path(__file__).resolve().parents[3]))
        for option in ('inputs', 'stu-weights', 'med-weights', 'official-root', 'dependency-root',
                       'chd-root', 'image-root'):
            child.add_argument('--'+option)
        child.add_argument('--proxy', help='仅对本次准备指定代理；默认直连并排除旧代理设置')
        child.add_argument('--install-dependencies', action='store_true')
    args = parser.parse_args()
    root, code = Path(args.root).resolve(strict=True), Path(args.code).resolve(strict=True)
    if code != Path(__file__).resolve().parents[3]:
        raise ValueError('安装器实际源码与--code不同，请使用目标main入口')
    source = Path(args.inputs).resolve(strict=True) if args.inputs else root/'course/chd-segmentation/inputs'
    assets = dict(stunet=Path(args.stu_weights).resolve() if args.stu_weights else root/'data/pretrained/STU-Net/small_ep4k.model',
                  medsam2=existing_or_default(args.med_weights,
                      [root/'data/model_weights/MedSAM2_latest.pt', root/'data/model_weights/MedSAM2/MedSAM2_latest.pt'],
                      root/'data/model_weights/MedSAM2/MedSAM2_latest.pt'),
                  official=existing_or_default(args.official_root, [code/'vendor/MedSAM2'],
                                               root/'code/third-party/MedSAM2'/MED_COMMIT),
                  dependencies=Path(args.dependency_root).resolve() if args.dependency_root else root/'env/medsam2')
    roots = {key: Path(value).resolve() for key, value in [('chd1282', args.chd_root), ('imagechd', args.image_root)] if value}
    preview = install(root, code, source, assets, check=True, dataset_roots=roots)
    dependencies = dependency_status(code, assets['dependencies'])
    if dependencies['missing']:
        validate_asset_destination(root, code, source, assets['dependencies'], roots)
    preview['environment'] = dependencies
    preview['network'] = dict(mode='explicit_proxy' if args.proxy else 'direct', inherited_proxy=False)
    if args.command == 'check':
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0 if not preview['missing'] and not dependencies['missing'] and dependencies['torch_compatible'] else 2
    blocked = [row for row in preview['missing'] if row['preparation'] not in ('official_download', 'official_git', 'linux_environment')]
    if blocked or not dependencies['torch_compatible'] or dependencies['missing'] and not args.install_dependencies:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 2
    course = root/'course/chd-segmentation'
    course.mkdir(parents=True, exist_ok=True)
    with FileLock(course/'.model-assets.lock', timeout=0):
        print(json.dumps(dict(stage='model_dependencies', action='verify_or_install'), ensure_ascii=False), flush=True)
        prepare_dependencies(code, root, assets['dependencies'], args.proxy, args.install_dependencies)
        print(json.dumps(dict(stage='MedSAM2_source', commit=MED_COMMIT), ensure_ascii=False), flush=True)
        clone_official(assets['official'], args.proxy)
        print(json.dumps(dict(stage='MedSAM2_weight', bytes=156040129, sha256=MED_SHA), ensure_ascii=False), flush=True)
        download_file(MED_URL, assets['medsam2'], 156040129, MED_SHA, args.proxy)
        result = install(root, code, source, assets, dataset_roots=roots)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    current_code = Path(__file__).resolve().parents[3]
    loader_spec = util.spec_from_file_location('chd_model_install_guard', current_code/'update.py')
    guard = util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(guard)
    with guard.code_guard(current_code/'run.py'):
        raise SystemExit(main())
