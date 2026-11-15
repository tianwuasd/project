import argparse
import json
from pathlib import Path
import shutil
import os
import sys
import subprocess
from time import perf_counter
import unittest
from uuid import uuid4

import psutil
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT/'src'))

from chd_segmentation.artifacts.store import file_digest, read_json, write_json
from chd_segmentation.delivery.install import inspect_inputs, ensure_q_inputs, install


class MetadataWorkspace:
    def __init__(self, root, report, cleaner, inventory):
        self.root, self.report = Path(root).resolve(), Path(report).resolve()
        self.cleaner, self.inventory = cleaner, inventory
        self.token = uuid4().hex
        self.results = []

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=False)
        self.report.mkdir(parents=True, exist_ok=True)
        write_json(self.root/'owner.json', dict(token=self.token))
        self.started = perf_counter()
        return self

    def __exit__(self, kind, error, traceback):
        files = self.inventory(self.root)
        result = dict(state='passed' if error is None else 'failed', error=None if error is None else str(error),
                      experiments=self.results, seconds=perf_counter()-self.started,
                      peak_rss_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', None),
                      peak_process_group_rss_bytes=None, peak_vram_bytes=None,
                      measurement_note='父进程实际Windows峰值；子命令进程组见各自监督回执',
                      artifact_count=len(files), artifact_bytes=sum(row['bytes'] for row in files),
                      artifacts_before_cleanup=files, work_root=str(self.root))
        write_json(self.report/'acceptance.json', result)
        self.cleaner(self.root, self.root.parent, self.token, self.report)
        result['cleanup'] = dict(work_root_exists=self.root.exists())
        write_json(self.report/'acceptance.json', result)
        print(json.dumps({key: result[key] for key in ('state', 'seconds', 'artifact_count', 'artifact_bytes', 'cleanup')}, ensure_ascii=False))


def run(args):
    sys.path.insert(0, args.simulator)
    from common import clean_owned, files_under, supervise
    source = Path(args.inputs)
    baseline = {p.name: (file_digest(p), p.stat().st_mtime_ns) for p in source.iterdir() if p.is_file()}
    with MetadataWorkspace(args.work_root, args.report, clean_owned, files_under) as workspace:
        if args.linked_destination:
            server = workspace.root/'server'
            course = server/'course/chd-segmentation'
            protected = workspace.root/'protected-dataset'
            protected.mkdir()
            course.mkdir(parents=True)
            relative = {'inputs': 'inputs', 'q': 'inputs/q1282',
                'bound': 'inputs/CHD1282-ImageCHD', 'receipt': 'installations',
                'configs': 'configs'}[args.linked_destination]
            link = course/relative
            link.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(protected)],
                           check=True, stdout=subprocess.PIPE)
            assets = dict(stunet=Path(args.stu), medsam2=Path(args.med),
                          official=Path(args.official), dependencies=Path(args.dependencies))
            try:
                with unittest.TestCase().assertRaises(ValueError):
                    install(server, PROJECT, source, assets, check=True,
                        dataset_roots=dict(chd1282=protected, imagechd=Path(args.image_root)))
                assert not list(protected.iterdir())
                workspace.results.append(dict(linked_destination=relative, rejected_before_write=True,
                    original_inputs_unchanged=True, ct_voxels_read=0, optimizer_updates=0))
            finally:
                os.rmdir(link)
            return
        started = perf_counter()
        installation_source = source
        if args.nested_inputs:
            nested = workspace.root/'server/course/chd-segmentation/inputs'
            shutil.copytree(source, nested)
            ensure_q_inputs(nested, nested/'q1282')
            installation_source = nested
        info = inspect_inputs(source)
        assert info['source_identity'] == 'd98440f41e27242a808694f8008a31984356dd032941f088298447007715eab3'
        qdir = workspace.root/'q-generated'
        first = ensure_q_inputs(source, qdir)
        assert first['identity'] == 'c427ad74beb3a720159479aaf4a1fa4c2df7c846c0e6bfad416e6167171005d3'
        assert file_digest(qdir/'q_groups.json') == '5f2c5a82fbbb746adbbb919a93080f886b1115fd6c2d183b14a8a8b9a97dc820'
        stamps = {p.name: p.stat().st_mtime_ns for p in qdir.iterdir() if p.is_file()}
        assert ensure_q_inputs(source, qdir)['reused']
        assert stamps == {p.name: p.stat().st_mtime_ns for p in qdir.iterdir() if p.is_file()}
        assets = dict(stunet=Path(args.stu), medsam2=Path(args.med),
                      official=Path(args.official), dependencies=Path(args.dependencies))
        server = workspace.root/'server'
        server.mkdir(exist_ok=True)
        preview_stamps = {str(p.relative_to(server)): p.stat().st_mtime_ns for p in server.rglob('*') if p.is_file()}
        preview = install(server, PROJECT, installation_source, assets, check=True,
                          dataset_roots=dict(chd1282=Path(args.chd_root), imagechd=Path(args.image_root)))
        assert preview_stamps == {str(p.relative_to(server)): p.stat().st_mtime_ns for p in server.rglob('*') if p.is_file()}
        assert preview['manual_upload_files'] == []
        installed = install(server, PROJECT, installation_source, assets,
                            dataset_roots=dict(chd1282=Path(args.chd_root), imagechd=Path(args.image_root)))
        server_stamps = {str(p.relative_to(server)): p.stat().st_mtime_ns for p in server.rglob('*') if p.is_file()}
        again = install(server, PROJECT, installation_source, assets,
                        dataset_roots=dict(chd1282=Path(args.chd_root), imagechd=Path(args.image_root)))
        assert not again['created_files']
        assert server_stamps == {str(p.relative_to(server)): p.stat().st_mtime_ns for p in server.rglob('*') if p.is_file()}
        command = [sys.executable, '-B', '-m', 'chd_segmentation.delivery', 'check',
                   '--root', str(server), '--code', str(PROJECT), '--inputs', str(installation_source),
                   '--stu-weights', args.stu, '--med-weights', args.med,
                   '--official-root', args.official, '--dependency-root', args.dependencies,
                   '--chd-root', args.chd_root, '--image-root', args.image_root]
        env = dict(os.environ, PYTHONPATH=str(PROJECT/'src'))
        previous = dict(os.environ)
        os.environ.update(env)
        try:
            supervise(command, PROJECT, workspace.root, workspace.report/'cli-check', timeout=120)
        finally:
            os.environ.clear()
            os.environ.update(previous)
        assert '"state": "checked"' in (workspace.report/'cli-check/console.log').read_text(encoding='utf-8')
        shell_env = dict(env, CHD_ROOT=server.as_posix(), CHD_PYTHON=Path(sys.executable).as_posix())
        shell_env['PATH'] = str(Path(args.bash).parent)+os.pathsep+shell_env['PATH']
        # 安装入口自行指定root/code，其余参数明确复用实际材料。
        shell = [args.bash, '--noprofile', '--norc', (PROJECT/'scripts/CHD1282-Q-MedSAM2-20261115_install.sh').as_posix(),
                 '--inputs', installation_source.as_posix(), '--stu-weights', Path(args.stu).as_posix(),
                 '--med-weights', Path(args.med).as_posix(), '--official-root', Path(args.official).as_posix(),
                 '--dependency-root', Path(args.dependencies).as_posix(), '--chd-root', Path(args.chd_root).as_posix(),
                 '--image-root', Path(args.image_root).as_posix()]
        os.environ.update(shell_env)
        try:
            supervise(shell, PROJECT, workspace.root, workspace.report/'shell-install', timeout=120)
        finally:
            os.environ.clear()
            os.environ.update(previous)
        assert '"state": "installed"' in (workspace.report/'shell-install/console.log').read_text(encoding='utf-8')
        cfg_path = server/'course/chd-segmentation/configs/CHD1282-Q/20261115/base.yaml'
        cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
        assert cfg['accumulation'] == 4 and cfg['updates'] == 40000
        groups = read_json(Path(cfg['groups']))
        assert {key: len(value) for key, value in groups.items()} == dict(A=393, B=197, C=128, D=456)
        before_conflict = cfg_path.read_bytes()
        cfg['updates'] = 123
        cfg_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
        changed_stamp = cfg_path.stat().st_mtime_ns
        with unittest.TestCase().assertRaises(ValueError):
            install(server, PROJECT, installation_source, assets,
                    dataset_roots=dict(chd1282=Path(args.chd_root), imagechd=Path(args.image_root)))
        assert cfg_path.stat().st_mtime_ns == changed_stamp
        assert yaml.safe_load(cfg_path.read_text(encoding='utf-8'))['updates'] == 123
        assert baseline == {p.name: (file_digest(p), p.stat().st_mtime_ns) for p in source.iterdir() if p.is_file()}
        workspace.results.append(dict(kind='existing-private-metadata-and-real-assets',
            ct_voxels_read=0, optimizer_updates=0, formal_case_metadata=len(info['groups']['A']+info['groups']['B']+info['groups']['C']+info['groups']['D']),
            q_reproduced=True, preflight_no_writes=True, install=installed,
            repeated_without_rewrite=True, conflicting_config_preserved=True,
            original_inputs_unchanged=True, seconds=perf_counter()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'stu', 'med', 'official', 'dependencies', 'chd-root', 'image-root', 'work-root', 'report'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--nested-inputs', action='store_true')
    parser.add_argument('--linked-destination', choices=('inputs', 'q', 'bound', 'receipt', 'configs'))
    parser.add_argument('--bash', default='C:/msys64/usr/bin/bash.exe')
    parser.add_argument('--simulator', default='D:/projects/test')
    run(parser.parse_args())
