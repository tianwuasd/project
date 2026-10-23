"""Offline mirror/release tests using synthetic local Git repositories."""
import io
import os
import signal
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from aaoca_server import environment, mirrors, updater
from aaoca_server.panel import Panel
from aaoca_server.progress import bar, lines
from aaoca_server.state import Registry, read, write


class MirrorTests(unittest.TestCase):
    def test_defaults_and_strict_origin(self):
        self.assertEqual(mirrors.DEFAULTS['code'], ['gh-proxy', 'ghfast', 'gitclone', 'official'])
        for url in [*mirrors.CODE.values(), 'https://gh-proxy.org/' + mirrors.OFFICIAL]:
            self.assertTrue(mirrors.known_origin(url.removesuffix('.git') + '/'))
        for url in ['https://evil.test/' + mirrors.OFFICIAL, mirrors.OFFICIAL + '-wrong',
                    'https://github.com/other/project.git', 'https://user:secret@github.com/tianwuasd/project.git']:
            self.assertFalse(mirrors.known_origin(url))
        for order in [[], ['ghfast', 'ghfast'], ['unknown'], [1]]:
            with self.assertRaises(ValueError):
                mirrors.validate({**mirrors.DEFAULTS, 'code': order})

    def test_mirror_menu_saves_order_and_cancel_does_not_save(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / 'panel'
            answers = iter(['m', '1', 'gitclone ghfast official', 'w', 'q'])
            Panel(runtime, ask=lambda _: next(answers), emit=lambda _: None).run()
            self.assertEqual(mirrors.load(runtime)['code'], ['gitclone', 'ghfast', 'official'])
            before = (runtime / 'mirrors.json').read_bytes()
            answers = iter(['m', '2', 'aliyun tuna', 'b', 'q'])
            Panel(runtime, ask=lambda _: next(answers), emit=lambda _: None).run()
            self.assertEqual((runtime / 'mirrors.json').read_bytes(), before)

    def test_archive_rejects_traversal_links_and_missing_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            for index, (name, kind) in enumerate([('AAOCA/../escape', tarfile.REGTYPE),
                                                ('AAOCA/aaoca_server/linked.py', tarfile.SYMTYPE),
                                                ('AAOCA/aaoca_server/only.py', tarfile.REGTYPE)]):
                content = io.BytesIO()
                with tarfile.open(fileobj=content, mode='w') as archive:
                    item = tarfile.TarInfo(name); item.type = kind; item.linkname = '/etc/passwd'
                    archive.addfile(item, io.BytesIO(b''))
                with self.assertRaises(ValueError):
                    updater.extract_release(content.getvalue(), Path(temp) / str(index))
            self.assertFalse((Path(temp) / 'escape').exists())


@unittest.skipUnless(shutil.which('git'), 'Git required for offline repository integration')
class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.upstream = self.root / 'upstream'; self.upstream.mkdir()
        self.real_git = updater.git
        self.real_git(self.upstream, 'init', '-b', 'main')
        self.source = self.upstream / 'AAOCA'; self.source.mkdir()
        for name in updater.RELEASE_FILES:
            path = updater.CODE_ROOT / name
            if path.is_dir():
                shutil.copytree(path, self.source / name, ignore=shutil.ignore_patterns('__pycache__'))
            else:
                shutil.copy2(path, self.source / name)
        self.base = self.commit()
        # A depth-one checkout reproduces the documented server bootstrap.
        self.checkout = self.root / 'checkout'
        self.real_git(self.root, 'clone', '--depth=1', self.upstream.as_uri(), str(self.checkout))
        self.real_git(self.checkout, 'remote', 'set-url', 'origin', mirrors.OFFICIAL)
        self.code = self.checkout / 'AAOCA'
        self.registry = Registry(self.root / 'panel')
        self.real_git(self.upstream, 'config', 'uploadpack.allowFilter', 'true')
        self.real_git(self.upstream, 'config', 'uploadpack.allowAnySHA1InWant', 'true')
        self.addCleanup(patch.stopall)
        patch.object(updater, 'CODE_ROOT', self.code).start()
        patch.dict(mirrors.CODE, {'gh-proxy': self.upstream.as_uri(), 'ghfast': self.upstream.as_uri()}).start()
        self.official = None
        self.routes = []

        def git(root, *args, **kwargs):
            if args[0] == 'ls-remote':
                if self.official:
                    return (self.official + '\trefs/heads/main\n').encode()
                raise ValueError('synthetic official offline')
            if args[:2] == ('config', 'remote.origin.url'):
                self.routes.append(args[2])
            return self.real_git(root, *args, **kwargs)
        patch.object(updater, 'git', side_effect=git).start()

    def commit(self):
        self.real_git(self.upstream, 'add', '.')
        self.real_git(self.upstream, '-c', 'user.name=Synthetic', '-c', 'user.email=test@example.invalid',
                      'commit', '-m', 'synthetic update')
        return self.real_git(self.upstream, 'rev-parse', 'HEAD').decode().strip()

    def advance(self):
        (self.source / 'aaoca_server/version.txt').write_text('next version', encoding='utf-8')
        return self.commit()

    def update(self):
        return updater.update(self.registry, emit=lambda _: None)

    def test_same_version_offline_stops_after_first_mirror(self):
        result = self.update()
        self.assertFalse(result['changed'])
        self.assertIn('官方最新状态未确认', result['message'])
        self.assertEqual(len(self.routes), 1)
        self.assertEqual(read(self.registry.root / 'update-last.json')['state'], 'unchanged')
        self.assertFalse((self.registry.root / 'release.json').exists())

    def test_success_publishes_isolated_release_preserves_checkout_and_history(self):
        commit = self.advance(); self.official = commit
        self.registry.register({'name': 'legacy', 'output': str(self.root / 'old-output')})
        result = self.update()
        self.assertTrue(result['changed']); self.assertEqual(len(self.routes), 1)
        release = updater.managed_release(self.registry.root)
        self.assertEqual(release.name, commit)
        self.assertEqual((release / 'aaoca_server/version.txt').read_text(), 'next version')
        self.assertEqual(self.real_git(self.checkout, 'rev-parse', 'HEAD').decode().strip(), self.base)
        self.assertEqual(self.real_git(self.checkout, 'status', '--porcelain'), b'')
        self.assertEqual(self.real_git(self.checkout, 'remote', 'get-url', 'origin').decode().strip(), mirrors.OFFICIAL)
        entry = self.registry.entries()[0]
        backup = Path(entry['code_root'])
        self.assertEqual((backup / 'aaoca_preprocessing/config.py').read_bytes(),
                         (self.code / 'aaoca_preprocessing/config.py').read_bytes())
        self.assertEqual(entry['python'], sys.executable)
        with self.assertRaises(ValueError): updater.ensure_current(self.registry.root)
        with patch.object(updater, 'CODE_ROOT', release):
            self.assertFalse(self.update()['changed'])

    def test_unavailable_first_route_falls_back_and_then_stops(self):
        commit = self.advance(); self.official = commit
        mirrors.CODE['gh-proxy'] = (self.root / 'missing.git').as_uri()
        self.assertTrue(self.update()['changed'])
        self.assertEqual(len(self.routes), 2)
        report = read(self.registry.root / 'update-last.json')
        self.assertEqual([x['state'] for x in report['attempts']], ['failed', 'verified'])

    def test_bad_archive_is_failure_and_next_valid_route_is_used(self):
        self.advance()
        original = updater.extract_release
        calls = []
        def extract(data, target):
            calls.append(target)
            if len(calls) == 1:
                raise ValueError('synthetic corrupt archive')
            return original(data, target)
        with patch.object(updater, 'extract_release', side_effect=extract):
            self.assertTrue(self.update()['changed'])
        self.assertEqual(len(self.routes), 2)

    def test_stale_mirror_is_rejected_when_official_is_known(self):
        self.official = 'f' * 40
        mirrors.save(self.registry.root, {**mirrors.DEFAULTS, 'code': ['gh-proxy']})
        with self.assertRaises(ValueError): self.update()
        self.assertFalse((self.registry.root / 'release.json').exists())
        self.assertEqual(read(self.registry.root / 'update-last.json')['state'], 'failed')

    def test_local_edits_or_active_jobs_prevent_network(self):
        marker = self.code / 'local-user-file.txt'; marker.write_text('keep')
        with self.assertRaises(ValueError): self.update()
        self.assertEqual(marker.read_text(), 'keep'); self.assertEqual(self.routes, [])
        marker.unlink()
        self.registry.register({'name': 'busy', 'output': str(self.root / 'job')})
        with patch('aaoca_server.state.job_status', return_value={'alive': True}):
            with self.assertRaises(RuntimeError): self.update()
        self.assertEqual(self.routes, [])

    def test_different_repository_is_rejected(self):
        self.real_git(self.checkout, 'remote', 'set-url', 'origin', 'https://github.com/other/project.git')
        with self.assertRaises(ValueError): self.update()
        self.assertEqual(self.routes, [])

    def test_launcher_follows_published_pointer_without_dependencies(self):
        self.advance(); self.update()
        release = updater.managed_release(self.registry.root)
        (release / 'start_aaoca_server.py').write_text('print("synthetic release loaded")\n')
        result = subprocess.run([sys.executable, '-S', str(self.code / 'start_aaoca_server.py'),
                                 '--runtime', str(self.registry.root), '--status'], capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b'synthetic release loaded', result.stdout)

    def test_pointer_write_failure_can_be_retried_without_overwriting_release(self):
        commit = self.advance()
        original = updater.write
        def fail_pointer(path, value):
            if path.name == 'release.json':
                raise OSError('synthetic disk failure during pointer publication')
            original(path, value)
        with patch.object(updater, 'write', side_effect=fail_pointer):
            with self.assertRaises(OSError): self.update()
        self.assertFalse((self.registry.root / 'release.json').exists())
        path = self.registry.root / 'releases' / commit / 'start_aaoca_server.py'
        before = path.stat().st_mtime_ns
        self.assertTrue(self.update()['changed'])
        self.assertEqual(path.stat().st_mtime_ns, before)


class InstallAndProgressTests(unittest.TestCase):
    def test_installer_stops_after_success_and_rejects_existing_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / 'env'
            with patch.object(environment, 'command', return_value=b'installed') as run:
                environment.install(destination, Path(temp) / 'panel', emit=lambda _: None)
                pip = [c for c in run.call_args_list if 'pip' in c.args[0]]
                self.assertEqual(len(pip), 1)
                self.assertIn(mirrors.PYPI['tuna'], pip[0].args[0])
                with self.assertRaises(ValueError): environment.install(destination, Path(temp))
            self.assertEqual(read(destination / 'install-report.json')['state'], 'installed')

    def test_installer_failure_uses_next_index_only(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []
            def run(args, **kwargs):
                if 'pip' in args:
                    calls.append(args[args.index('--index-url') + 1])
                    if len(calls) == 1: raise ValueError('synthetic unreachable')
                return b'ok'
            with patch.object(environment, 'command', side_effect=run):
                environment.install(Path(temp) / 'env', Path(temp), emit=lambda _: None)
            self.assertEqual(calls, [mirrors.PYPI['tuna'], mirrors.PYPI['sjtu']])

    def test_resume_uses_frozen_interpreter_and_code(self):
        with patch.object(updater, 'command', return_value=b'ok') as run:
            entry = {'code_root': str(updater.CODE_ROOT), 'python': sys.executable, 'output': '/synthetic/job'}
            updater.resume_saved(entry)
            self.assertEqual(run.call_args.args[0][0], sys.executable)
            self.assertEqual(run.call_args.args[0][-2:], [str(updater.CODE_ROOT), '/synthetic/job'])

    def test_progress_counts_finished_failures_separately_and_shows_slices(self):
        text = '\n'.join(lines({'total': 10, 'completed': 3, 'reused': 2, 'failed': 1,
                                'active': {'case_a': {'stage': 'compute', 'completed_slices': 4, 'total_slices': 8}}}))
        self.assertIn('60.0%', text); self.assertIn('(6/10)', text)
        self.assertIn('50.0%', text); self.assertIn('重采样', text)
        self.assertIn('失败 1', text); self.assertIn('不代表全部成功', text)
        self.assertNotIn('100.0%', bar(0, 0))
        self.assertIn('100.0%', bar(20, 10))

    def test_progress_ctrl_c_never_stops_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp)
            output = runtime / 'out'
            write(runtime / 'history.json', [{'name': 'synthetic', 'output': str(output)}])
            messages = []
            panel = Panel(runtime, emit=messages.append)
            with patch.object(panel, 'show_status'), \
                 patch('aaoca_server.panel.job_status', return_value={'alive': True}), \
                 patch('aaoca_server.panel.time.sleep', side_effect=KeyboardInterrupt):
                panel.watch_progress()
            self.assertFalse((runtime / 'out.runtime/stop.request').exists())
            self.assertTrue(any('后台任务继续运行' in message for message in messages))

    def test_command_timeout_is_bounded(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            updater.command([sys.executable, '-c', 'import time; time.sleep(30)'], timeout=.2)

    def test_interrupt_during_process_creation_is_deferred_until_child_is_owned(self):
        child = Mock(pid=123456)
        child.communicate.return_value = (b'', b'')
        original = signal.getsignal(signal.SIGINT)
        def spawn(*args, **kwargs):
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            return child
        with patch.object(updater.subprocess, 'Popen', side_effect=spawn), \
             patch.object(updater.subprocess, 'run'), patch.object(updater.os, 'killpg', create=True):
            with self.assertRaises(KeyboardInterrupt): updater.command(['synthetic'])
        child.communicate.assert_called_once_with(timeout=3)
        self.assertEqual(signal.getsignal(signal.SIGINT), original)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux process-group lifecycle')
    def test_timeout_removes_helper_that_closed_pipes_and_ignored_term(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / 'helper.pid'
            helper = ('import os,signal,sys,time; from pathlib import Path; '
                      'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                      'Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)')
            parent = ('import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2]], '
                      'stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(30)')
            with self.assertRaises(subprocess.TimeoutExpired):
                updater.command([sys.executable, '-c', parent, helper, str(marker)], timeout=1)
            self.assertTrue(marker.exists())
            pid = int(marker.read_text())
            def running():
                status = Path(f'/proc/{pid}/stat')
                try:
                    return status.read_text().split(') ', 1)[1].split()[0] != 'Z'
                except FileNotFoundError:
                    return False
            try:
                for _ in range(20):
                    if not running(): break
                    time.sleep(.05)
                self.assertFalse(running(), 'owned helper survived command timeout')
            finally:
                if running(): os.kill(pid, signal.SIGKILL)


if __name__ == '__main__':
    unittest.main()
