import argparse
import hashlib
from pathlib import Path
import stat
import subprocess
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT/'src'))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import file_digest
from chd_segmentation.delivery.assets import download_file, clone_official, MED_COMMIT


def run(args):
    original = Path(args.official)
    content = subprocess.run(['git', '-C', str(original), 'show', MED_COMMIT+':LICENSE'],
                             check=True, capture_output=True).stdout
    size, sha = len(content), hashlib.sha256(content).hexdigest()
    url = f'https://raw.githubusercontent.com/bowang-lab/MedSAM2/{MED_COMMIT}/LICENSE'
    with SmokeWorkspace(args.work_root, args.report) as workspace:
        target = workspace.root/'LICENSE'
        first = download_file(url, target, size, sha, args.proxy)
        stamp = target.stat().st_mtime_ns
        assert target.read_bytes() == content
        assert download_file(url, target, size, sha, args.proxy)['reused']
        assert stamp == target.stat().st_mtime_ns
        with unittest.TestCase().assertRaises(ValueError):
            download_file(url, workspace.root/'wrong-digest', size, '0'*64, args.proxy)
        assert not (workspace.root/'wrong-digest').exists()
        assert not (workspace.root/'wrong-digest.pending').exists()
        clone = workspace.root/'MedSAM2'
        fetched = clone_official(clone, args.proxy)
        stamps = {str(p.relative_to(clone)): p.stat().st_mtime_ns for p in clone.rglob('*')
                  if p.is_file() and '.git' not in p.relative_to(clone).parts}
        assert clone_official(clone, args.proxy)['reused']
        assert stamps == {str(p.relative_to(clone)): p.stat().st_mtime_ns for p in clone.rglob('*')
                          if p.is_file() and '.git' not in p.relative_to(clone).parts}
        assert file_digest(clone/'LICENSE') == sha
        workspace.results.append(dict(ct_voxels_read=0, optimizer_updates=0, download=first,
            official_clone=fetched, unchanged_files_reused=True, bad_digest_rejected=True,
            proxy=args.proxy, full_weight_download_not_repeated=True))
        # 新建Git对象默认只读；相关子进程均已返回，仅释放本次测试文件属性。
        for path in clone.rglob('*'):
            if path.is_file():
                path.chmod(path.stat().st_mode | stat.S_IWRITE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('official', 'proxy', 'work-root', 'report'):
        parser.add_argument('--'+name, required=True)
    run(parser.parse_args())
