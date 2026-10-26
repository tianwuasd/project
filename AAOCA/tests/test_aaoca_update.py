"""Offline private-route/release tests using synthetic local Git repositories."""
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
from types import SimpleNamespace
from unittest.mock import Mock, patch

from aaoca_server import environment, mirrors, network, updater
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
            answers = iter(['m', '2', 'sjtu tuna', 'w', 'q'])
            Panel(runtime, ask=lambda _: next(answers), emit=lambda _: None).run()
            self.assertEqual(mirrors.load(runtime)['pypi'], ['sjtu', 'tuna'])
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
        self.official = self.base
        self.routes = []

        def git(root, *args, **kwargs):
            if args[:2] == ('ls-remote', '--get-url'):
                return self.real_git(root, *args, **kwargs)
            if args[0] == 'ls-remote':
                if self.official:
                    return (self.official + '\trefs/heads/main\n').encode()
                raise ValueError('synthetic official offline')
            if args[:2] == ('config', 'remote.origin.url'):
                self.routes.append(args[2])
                args = (*args[:2], self.upstream.as_uri())
            return self.real_git(root, *args, **kwargs)
        patch.object(updater, 'git', side_effect=git).start()

    def commit(self):
        self.real_git(self.upstream, 'add', '.')
        self.real_git(self.upstream, '-c', 'user.name=Synthetic', '-c', 'user.email=test@example.invalid',
                      'commit', '-m', 'synthetic update')
        return self.real_git(self.upstream, 'rev-parse', 'HEAD').decode().strip()

    def advance(self):
        (self.source / 'aaoca_server/version.txt').write_text('next version', encoding='utf-8')
        self.official = self.commit()
        return self.official

    def update(self):
        return updater.update(self.registry, emit=lambda _: None)

    def test_same_version_official_stops_after_one_route(self):
        result = self.update()
        self.assertFalse(result['changed'])
        self.assertIn('已与官方最新版本一致', result['message'])
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

    def test_official_auth_failure_never_uses_legacy_mirrors(self):
        self.official = None
        mirrors.save(self.registry.root, mirrors.DEFAULTS)
        with self.assertRaises(ValueError): self.update()
        self.assertEqual(self.routes, [])
        self.assertFalse((self.registry.root / 'release.json').exists())
        self.assertEqual(read(self.registry.root / 'update-last.json')['state'], 'failed')

    def test_bad_archive_stops_without_public_fallback(self):
        self.advance()
        with patch.object(updater, 'extract_release', side_effect=ValueError('synthetic corrupt archive')):
            with self.assertRaises(ValueError): self.update()
        self.assertEqual(self.routes, [network.HTTPS_UPSTREAM])
        self.assertFalse((self.registry.root / 'release.json').exists())

    def test_download_has_separate_limits_real_progress_and_stage_receipts(self):
        self.advance()
        self.assertTrue(self.update()['changed'])
        calls = updater.git.call_args_list
        queries = [c for c in calls if c.args[1] == 'ls-remote' and '--get-url' not in c.args]
        fetches = [c for c in calls if c.args[1] == 'fetch']
        archives = [c for c in calls if c.args[1] == 'archive']
        self.assertEqual(queries[0].kwargs['timeout'], 30)
        self.assertTrue(all(100 < c.kwargs['timeout'] <= 900 for c in fetches))
        self.assertTrue(all('--progress' in c.args for c in fetches))
        self.assertEqual(archives[0].kwargs['timeout'], 180)
        report = read(self.registry.root / 'update-last.json')
        self.assertTrue(Path(report['log']).is_file())
        self.assertTrue(report['stages'])
        self.assertTrue(all(s['seconds'] >= 0 for s in report['stages']))

    def test_bare_archive_ignores_tree_and_config_attributes_for_whitelisted_code(self):
        (self.upstream / '.gitattributes').write_text('*.py export-ignore\n')
        (self.source / '.gitattributes').write_text('aaoca_server/panel.py export-ignore\n')
        other = self.upstream / 'unselected'; other.mkdir()
        (other / '.gitattributes').write_text('# unrelated attribute object\n')
        self.official = self.commit()
        attributes = self.root / 'configured-attributes'
        attributes.write_text('* export-ignore\n')
        original = updater.git.side_effect
        def configure_archive(root, *args, **kwargs):
            if args[0] == 'archive':
                self.real_git(root, 'config', 'core.attributesfile', str(attributes))
            return original(root, *args, **kwargs)
        with patch.object(updater, 'git', side_effect=configure_archive):
            self.assertTrue(self.update()['changed'])
        release = updater.managed_release(self.registry.root)
        expected = self.real_git(self.upstream, 'show', 'HEAD:AAOCA/aaoca_server/panel.py')
        self.assertEqual((release / 'aaoca_server/panel.py').read_bytes(), expected)

    def test_archive_with_unfetched_attributes_succeeds_without_remote_access(self):
        (self.source / '.gitattributes').write_text('# intentionally unfetched parent attribute\n')
        other = self.upstream / 'unselected'; other.mkdir()
        (other / '.gitattributes').write_text('# intentionally unfetched unrelated attribute\n')
        self.official = self.commit()
        original = updater.git.side_effect
        run = updater.download.run
        environments = []
        def trace_run(args, **kwargs):
            if 'archive' in args:
                environments.append(kwargs['env'])
            return run(args, **kwargs)
        def offline_archive(root, *args, **kwargs):
            if args[0] == 'archive':
                self.real_git(root, 'config', 'remote.origin.url', (self.root / 'missing-remote').as_uri())
            return original(root, *args, **kwargs)
        with patch.object(updater, 'git', side_effect=offline_archive), \
             patch.object(updater.download, 'run', side_effect=trace_run):
            self.assertTrue(self.update()['changed'])
        self.assertEqual(environments[0]['GIT_ALLOW_PROTOCOL'], '')
        self.assertEqual(environments[0]['GIT_ATTR_NOSYSTEM'], '1')
        cache = Path(read(self.registry.root / 'update-last.json')['cache'])
        present = self.real_git(cache, 'cat-file', '--batch-all-objects', '--batch-check=%(objectname)').splitlines()
        missing_attr = self.real_git(self.upstream, 'rev-parse', 'HEAD:AAOCA/.gitattributes').strip()
        self.assertNotIn(missing_attr, present)  # The archive did not fetch it.

    def test_missing_code_blob_fails_locally_without_hidden_fetch_or_publication(self):
        self.advance()
        original = updater.git.side_effect
        def omit_code_blobs(root, *args, **kwargs):
            if args[0] == 'fetch' and '--filter=blob:none' not in args:
                return b''  # Simulate an incomplete object download.
            return original(root, *args, **kwargs)
        with patch.object(updater, 'git', side_effect=omit_code_blobs):
            with self.assertRaisesRegex(ValueError, '本地归档.*隐式联网已禁止'):
                self.update()
        self.assertFalse((self.registry.root / 'release.json').exists())
        report = read(self.registry.root / 'update-last.json')
        self.assertEqual(report['stages'][-1]['error_kind'], 'git')

    def test_failed_download_cache_is_reused_on_retry(self):
        self.advance()
        with patch.object(updater, 'extract_release', side_effect=ValueError('synthetic corrupt archive')):
            with self.assertRaises(ValueError): self.update()
        first = read(self.registry.root / 'update-last.json')
        cache = Path(first['cache'])
        marker = cache / 'preserved-marker'; marker.touch()
        self.assertTrue(self.update()['changed'])
        self.assertEqual(read(self.registry.root / 'update-last.json')['cache'], str(cache))
        self.assertTrue(marker.exists())

    def test_fetch_timeout_names_stage_preserves_current_pointer_and_logs(self):
        self.advance()
        original = updater.git.side_effect
        def fail_fetch(root, *args, **kwargs):
            if args[0] == 'fetch':
                raise subprocess.TimeoutExpired(['git', 'fetch'], kwargs['timeout'])
            return original(root, *args, **kwargs)
        with patch.object(updater, 'git', side_effect=fail_fetch):
            with self.assertRaisesRegex(ValueError, '对象下载.*超时'):
                self.update()
        report = read(self.registry.root / 'update-last.json')
        self.assertEqual(report['stages'][-1]['error_kind'], 'timeout')
        self.assertTrue(Path(report['cache']).exists())
        self.assertFalse((self.registry.root / 'release.json').exists())

    def test_shared_deadline_applies_to_cached_object_inspection(self):
        self.advance()
        with patch.object(updater, 'extract_release', side_effect=ValueError('synthetic archive failure')):
            with self.assertRaises(ValueError): self.update()
        clock = [0.0]
        original = updater.git.side_effect
        def slow_cache(root, *args, **kwargs):
            result = original(root, *args, **kwargs)
            if args[0] == 'cat-file':
                clock[0] += 901
            return result
        with patch.object(updater, 'time', SimpleNamespace(monotonic=lambda: clock[0])), \
             patch.object(updater, 'git', side_effect=slow_cache):
            with self.assertRaisesRegex(ValueError, '对象下载.*超时'):
                self.update()
        self.assertFalse((self.registry.root / 'release.json').exists())

    def test_moving_remote_is_rejected_when_confirmed_commit_changes(self):
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

    def test_checkout_local_url_rewrite_prevents_any_network_attempt(self):
        self.real_git(self.checkout, 'config', 'url.https://mirror.invalid/.insteadOf', 'https://github.com/')
        with self.assertRaisesRegex(ValueError, '重写'):
            self.update()
        self.assertEqual(self.routes, [])

    def test_bootstrap_upgrades_existing_release_and_backs_up_actual_old_code(self):
        from aaoca_server.bootstrap import upgrade
        self.advance()
        self.update()
        old = updater.managed_release(self.registry.root)
        self.registry.register({'name': 'old-release-job', 'output': str(self.root / 'old-job'),
                                'code_root': str(old), 'python': sys.executable})
        (self.source / 'aaoca_server/version.txt').write_text('version two')
        self.official = self.commit()
        self.assertTrue(upgrade(self.registry.root, self.checkout, emit=lambda _: None)['changed'])
        self.assertEqual(updater.managed_release(self.registry.root).name, self.official)
        saved = Path(self.registry.entries()[0]['code_root'])
        self.assertEqual((saved / 'aaoca_server/version.txt').read_text(), 'next version')
        self.assertEqual(self.real_git(self.checkout, 'rev-parse', 'HEAD').decode().strip(), self.base)

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
