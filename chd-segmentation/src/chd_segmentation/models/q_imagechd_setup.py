import argparse
import json
from pathlib import Path
import random
import subprocess
import sys


BUDGETS = dict(base=40000, localizer=5000, local=20000)


def split_groups(manifest, seed):
    if manifest['dataset'] != 'imagechd':
        raise ValueError('需要ImageCHD病例清单')
    pools = {role: [] for role in ('train', 'validation', 'test')}
    seen = set()
    for row in manifest['cases']:
        name = row['case_id']
        roles = set(row['roles']) - {'predict'}
        if name in seen or len(roles) != 1 or not roles <= pools.keys():
            raise ValueError('病例重复或角色不明确：' + name)
        pools[next(iter(roles))].append(name)
        seen.add(name)
    train = sorted(pools['train'])
    random.Random(seed).shuffle(train)
    boundary = len(train) * 2 // 3
    groups = dict(A=sorted(train[:boundary]), B=sorted(train[boundary:]),
                  C=sorted(pools['validation']), D=sorted(pools['test']))
    if not all(groups.values()):
        raise ValueError('Q需要非空A/B/C/D四组')
    return groups


def stage_config(root, directory, pretrained, stage, seed):
    run = root / 'course/chd-segmentation/runs/ImageCHD-Q-R2_fold0'
    return dict(model='q_route1', run_dir=str(run),
                work_root=str(root / 'course/chd-segmentation'), device='cuda',
                precision='fp32', seed=seed, technical=False, updates=BUDGETS[stage],
                accumulation=4, patch=128, local_size=128, learning_rate=.0002,
                pretrained=None if stage == 'localizer' else str(pretrained),
                groups=str(directory / 'groups.json'),
                parents={key: str(run / key / 'best.pt') for key in ('base', 'localizer')}
                if stage == 'local' else {},
                execution=dict(gpu_count=2, min_gpus=2, workers=0, share=False,
                               wait_seconds=None))


def setup(root, code, seed):
    sys.path.insert(0, str(code / 'src'))
    from filelock import FileLock
    from chd_segmentation.artifacts.store import read_json, write_json
    from chd_segmentation.models.offline import validate_model_storage
    from chd_segmentation.models.workflow import options, validated_q_groups

    course = root / 'course/chd-segmentation'
    frozen = read_json(course / 'runs/ImageCHD-STUNet-20261115_fold0/run.json')
    prepared = frozen['prepared']
    if (prepared['dataset'], prepared['scope'], prepared['fold']) != ('imagechd', 'formal', 0):
        raise ValueError('需要正式ImageCHD fold0训练记录')
    groups = split_groups(prepared['manifest'], seed)
    actual = validate_model_storage(dict(model='q_route1'), prepared)
    pretrained = Path(frozen['config']['pretrained'])
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    directory = course / 'configs/ImageCHD-Q-R2_fold0'
    outputs = {'groups.json': groups, 'prepared.json': prepared}
    outputs.update({stage + '.json': stage_config(root, directory, pretrained, stage, seed)
                    for stage in BUDGETS})
    outputs['input.json'] = dict(stunet_run=str(course / 'runs/ImageCHD-STUNet-20261115_fold0'),
                                prepared_identity=prepared['identity'], actual_recipe=actual,
                                group_seed=seed, training_partition='A:B=2:1',
                                preprocessing='reuse_existing_stunet_arrays')
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(directory / 'setup.lock', timeout=0):
        for name, value in outputs.items():
            path = directory / name
            if path.exists() and read_json(path) != value:
                raise ValueError('已有Q配置与本次输入不同：' + str(path))
        for name, value in outputs.items():
            if not (directory / name).exists():
                write_json(directory / name, value)
        for stage in BUDGETS:
            validated_q_groups(options(directory / (stage + '.json')), prepared)
    print(json.dumps(dict(config_directory=str(directory), groups={k: len(v) for k, v in groups.items()},
                          reused_preprocessed_root=prepared['storage']['root'], budgets=BUDGETS),
                     ensure_ascii=False), flush=True)
    return directory


def run_stages(root, code, directory):
    from chd_segmentation.artifacts.store import read_json

    run = root / 'course/chd-segmentation/runs/ImageCHD-Q-R2_fold0'
    for stage, updates in BUDGETS.items():
        command = [sys.executable, '-u', '-B', str(code / 'run.py'), 'models', 'train',
                   '--config', str(directory / (stage + '.json')),
                   '--prepared', str(directory / 'prepared.json'), '--stage', stage,
                   '--device', 'cuda', '--gpus', '2', '--batch', '4',
                   '--no-share-gpu', '--wait-seconds', 'forever']
        if (run / stage / 'latest.pt').is_file():
            command.append('--resume')
        subprocess.run(command, check=True)
        if read_json(run / stage / 'summary.json')['update'] != updates:
            raise RuntimeError('当前阶段尚未完成，停止后续阶段：' + stage)


def stop(root, code):
    import os
    import psutil

    owner = psutil.Process().username()
    directory = root / 'course/chd-segmentation/configs/ImageCHD-Q-R2_fold0'
    configs = {str(directory / (stage + '.json')) for stage in BUDGETS}
    launchers = {str(root / 'course/chd-segmentation/bootstrap/q_imagechd_setup.py'),
                 str(code / 'src/chd_segmentation/models/q_imagechd_setup.py'),
                 'chd_segmentation.models.q_imagechd_setup'}
    selected = {}
    for process in psutil.process_iter(['pid', 'username', 'cmdline'], ad_value=None):
        argv = process.info['cmdline'] or []
        if process.pid == os.getpid() or process.info['username'] != owner:
            continue
        launcher = bool(launchers.intersection(argv)) and '--run' in argv
        training = (str(code / 'run.py') in argv and 'models' in argv and 'train' in argv
                    and '--config' in argv and argv.index('--config') + 1 < len(argv)
                    and argv[argv.index('--config') + 1] in configs)
        if launcher or training:
            selected[process.pid] = process
            for child in process.children(recursive=True):
                selected[child.pid] = child
    for process in selected.values():
        if process.is_running():
            process.terminate()
    _, living = psutil.wait_procs(list(selected.values()), timeout=5)
    for process in living:
        if process.is_running():
            process.kill()
    _, living = psutil.wait_procs(living, timeout=5)
    if living:
        raise RuntimeError('Q进程尚未退出：' + str([process.pid for process in living]))
    print(json.dumps(dict(stopped=sorted(selected))), flush=True)


def main():
    parser = argparse.ArgumentParser(description='复用ImageCHD STU fold0数据启动Q-r2')
    parser.add_argument('--root', type=Path, default=Path('/data4/zhangruichen'))
    parser.add_argument('--code', type=Path, default=Path('/data4/zhangruichen/code/chd-segmentation'))
    parser.add_argument('--seed', type=int, default=20261111)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    if args.run and args.stop:
        parser.error('--run与--stop不能同时使用')
    if args.stop:
        stop(args.root.resolve(), args.code.resolve())
        return
    directory = setup(args.root.resolve(), args.code.resolve(), args.seed)
    if args.run:
        run_stages(args.root.resolve(), args.code.resolve(), directory)


if __name__ == '__main__':
    main()
