import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
from time import perf_counter
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import SmokeWorkspace

spec = importlib.util.spec_from_file_location('assistant2_updater', ROOT/'update.py')
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)
BRANCH = 'main'


def git(path, *args, data=None):
    return subprocess.run(['git', '-C', str(path), '-c', 'user.name=Codex',
        '-c', 'user.email=codex@local', '-c', 'gc.auto=0', *args], input=data,
        stdout=subprocess.PIPE, timeout=60, check=True).stdout


def commit(origin, readme, parent=None):
    entries = {}
    for name in ('run.py', 'update.py', 'setup_env.py', 'pyproject.toml'):
        entries[name] = git(origin, 'hash-object', '-w', '--stdin', data=(ROOT/name).read_bytes()).decode().strip()
    entries['README.md'] = git(origin, 'hash-object', '-w', '--stdin', data=readme).decode().strip()
    tree = git(origin, 'mktree', data=''.join(
        f'100644 blob {blob}\t{name}\n' for name, blob in sorted(entries.items())).encode()).decode().strip()
    root = git(origin, 'mktree', data=f'040000 tree {tree}\tchd-segmentation\n'.encode()).decode().strip()
    parents = () if parent is None else ('-p', parent)
    return git(origin, 'commit-tree', root, *parents, data=b'Actual updater source snapshot\n').decode().strip()


def run(args):
    checks, check = [], TestCase()
    with SmokeWorkspace(args.work_root, args.report) as workspace:
        started, peaks = perf_counter(), []
        try:
            check.assertEqual(updater.BRANCH, 'main')
            origin = workspace.root/'origin.git'
            origin.mkdir()
            git(origin, 'init', '--bare', '--template=')
            git(origin, 'config', 'uploadpack.allowFilter', 'true')
            git(origin, 'config', 'uploadpack.allowAnySHA1InWant', 'true')
            old_readme = git(ROOT, 'show', '298a75a50e18117bd62c741ebbc328b89dc8e7ad:README.md')
            first = commit(origin, old_readme)
            second = commit(origin, (ROOT/'README.md').read_bytes(), first)
            git(origin, 'update-ref', 'refs/heads/main', first)
            git(origin, 'update-ref', 'refs/heads/codex/chd-assistant2-offline-preprocess-20261115', first)
            url = origin.as_uri()
            deployment, protected = workspace.root/'assistant2-repo', workspace.root/'assistant1-repo'
            for branch, path in ((BRANCH, deployment), ('main', protected)):
                subprocess.run(['git', '-c', 'init.templateDir=', 'clone', '--no-checkout',
                    '--no-tags', '--filter=blob:none', '--branch', branch, url, str(path)],
                    stdout=subprocess.PIPE, timeout=60, check=True)
            keep = protected/'keep.local.json'
            updater.write_json(keep, dict(scope='protected independent main checkout'))
            protected_before = (git(protected, 'rev-parse', 'HEAD'), keep.read_bytes(), keep.stat().st_mtime_ns)
            git(origin, 'update-ref', 'refs/heads/'+BRANCH, second, first)
            repo = updater.GitRepository(deployment, expected_urls={url})
            check.assertEqual(repo.remote_head(), second, '更新必须读取main')
            check.assertEqual(git(origin, 'rev-parse', 'refs/heads/codex/chd-assistant2-offline-preprocess-20261115').decode().strip(), first)
            checks.append('remote target follows main while the frozen assistant2 branch remains unchanged')
            initial = updater.GitRepository(protected, expected_urls={url})
            git(protected, 'update-ref', 'refs/heads/other', first)
            git(protected, 'symbolic-ref', 'HEAD', 'refs/heads/other')
            for force in (False, True):
                with check.assertRaises(ValueError):
                    initial.validate(force=force)
            git(protected, 'symbolic-ref', 'HEAD', 'refs/heads/main')
            checks.append('another local branch is rejected including force mode')
            runtime = workspace.root/'assistant2-course'
            choice = repo.choose_source([url], expected=second, timeout=5, official=url)
            check.assertEqual(choice, second)
            result = updater.synchronize(repo, runtime, second)
            check.assertEqual(result['sha'], second)
            check.assertEqual(result['branch'], BRANCH)
            check.assertEqual(result['origin_branch'], second)
            check.assertEqual(repo.run('rev-parse', 'refs/remotes/origin/'+BRANCH).decode().strip(), second)
            code = deployment/'chd-segmentation'
            check.assertEqual((code/'README.md').read_bytes(), (ROOT/'README.md').read_bytes())
            check.assertEqual(repo.remote_head(), second)
            peaks.append(sum(1 for path in workspace.root.rglob('*') if path.is_file()))
            checks.append('actual Git synchronization applies and verifies the fixed branch')
            with updater.code_guard(code/'run.py'):
                with check.assertRaises(OSError):
                    updater.synchronize(repo, runtime, second)
            times = {p.name: p.stat().st_mtime_ns for p in code.iterdir()}
            repeated = updater.synchronize(repo, runtime, second)
            check.assertEqual(repeated['sha'], second)
            check.assertEqual(times, {p.name: p.stat().st_mtime_ns for p in code.iterdir()})
            info = json.loads(subprocess.run([sys.executable, '-B', str(code/'run.py'), '--code-info'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=True).stdout)
            check.assertEqual(info['sha'], second)
            check.assertEqual(Path(info['code']), code)
            check.assertFalse(info['modified'])
            checks.append('active code lock blocks update; repeated update retains files and actual entry reads target SHA')
            third = commit(origin, (ROOT/'README.md').read_bytes(), second)
            git(origin, 'update-ref', 'refs/heads/'+BRANCH, third, second)
            with check.assertRaises(ValueError):
                updater.synchronize(repo, runtime, second)
            check.assertEqual(repo.run('rev-parse', 'HEAD').decode().strip(), second)
            check.assertEqual((git(protected, 'rev-parse', 'HEAD'), keep.read_bytes(), keep.stat().st_mtime_ns), protected_before)
            check.assertEqual(git(origin, 'rev-parse', 'refs/heads/main').decode().strip(), third)
            checks.append('remote advancement rejects the old target; independent main checkout remains unchanged')
            files = [p for p in workspace.root.rglob('*') if p.is_file()]
            peaks.append(len(files))
            check.assertLessEqual(max(peaks), 200)
            check.assertLessEqual(sum(p.stat().st_size for p in files), 8*2**30)
            workspace.results.append(dict(checks=checks, fixture_commits=[first, second, third],
                source_files=5, network='local file Git source; no outbound access',
                protected_head=first, actual_entry=info, stage_sampled_peak_files=max(peaks),
                seconds=perf_counter()-started, optimizer_updates=0,
                source_sha256=hashlib.sha256((ROOT/'update.py').read_bytes()).hexdigest()))
        finally:
            # Git对象可以只读；只调整本次独占测试目录中的文件权限，交给工作区核对并清理。
            for path in workspace.root.rglob('*'):
                if path.is_file():
                    path.chmod(stat.S_IREAD | stat.S_IWRITE)


def inventory_run(args):
    with SmokeWorkspace(args.work_root, args.report) as workspace:
        origin = workspace.root/'inventory.git'
        subprocess.run(['git', '-c', 'init.templateDir=', 'clone', '--bare', '--shared',
            '--single-branch', str(ROOT), str(origin)], check=True, stdout=subprocess.PIPE, timeout=60)
        names = git(ROOT, 'ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')
        files = {name: git(origin, 'hash-object', '-w', '--stdin', data=(ROOT/name).read_bytes()).decode().strip()
                 for name in sorted(set(names)) if name and updater.allowed(name) and (ROOT/name).is_file()}
        def tree(prefix):
            entries = {}
            for name, blob in files.items():
                if name.startswith(prefix):
                    leaf, separator, _ = name[len(prefix):].partition('/')
                    entries[leaf] = ('040000 tree '+tree(prefix+leaf+'/')) if separator and leaf not in entries else entries.get(leaf, '100644 blob '+blob)
            return git(origin, 'mktree', '-z', data=b''.join(
                (value+'\t'+name).encode()+b'\0' for name,value in sorted(entries.items()))).decode().strip()
        root = git(origin, 'mktree', data=('040000 tree '+tree('')+'\tchd-segmentation\n').encode()).decode().strip()
        head = git(origin, 'commit-tree', root, data=b'Actual managed source inventory\n').decode().strip()
        repo = updater.GitRepository(origin, expected_urls={str(ROOT)})
        try:
            actual = repo.inventory(head)
            assert len(actual) > 200 and actual == files
            workspace.results.append(dict(source_files=len(actual), actual_git_inventory=True, optimizer_updates=0))
        finally:
            for path in workspace.root.rglob('*'):
                if path.is_file():
                    path.chmod(stat.S_IREAD | stat.S_IWRITE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-root', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    (inventory_run if args.inventory_only else run)(args)
