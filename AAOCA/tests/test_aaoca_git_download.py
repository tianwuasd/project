"""Real local subprocesses verify streaming and cancellation; no network required."""
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from aaoca_server import download
from aaoca_server import updater
from unittest.mock import Mock, patch


class DownloadTests(unittest.TestCase):
    def test_progress_reconstructs_numeric_fields_without_arbitrary_suffix(self):
        text = download.safe_progress('Receiving objects: 50% (1/2), Bearer abc123')
        self.assertEqual(text, 'Receiving objects: 50% (1/2)')
        text = download.safe_progress('remote: Receiving objects: 50% (1/2), 1.0 MiB | 2.0 MiB/s, abc123')
        self.assertEqual(text, 'remote: Receiving objects: 50% (1/2), 1.0 MiB | 2.0 MiB/s')

    def test_nonstreaming_git_sigterm_is_deferred_and_cleanup_is_protected(self):
        child = Mock(pid=123456)
        child.communicate.return_value = (b'', b'')
        before = signal.getsignal(signal.SIGTERM)
        def spawn(*args, **kwargs):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return child
        with patch.object(updater.subprocess, 'Popen', side_effect=spawn), \
             patch.object(updater.subprocess, 'run'), patch.object(updater.os, 'killpg', create=True):
            with self.assertRaises(KeyboardInterrupt): updater.command(['synthetic'])
        child.communicate.assert_called_once_with(timeout=3)
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)

    def test_carriage_return_progress_arrives_before_exit_and_stdout_stays_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'finished'
            log = Path(tmp) / 'download.log'
            messages = []
            def emit(message):
                messages.append((message, marker.exists()))
            script = ('import os,time,sys; from pathlib import Path; '
                      'os.write(2,b"Receiving objects: 50% (1/2), 1.0 KiB | 1.0 KiB/s\\r"); '
                      'os.write(1,b"\\x00archive\\xff"); time.sleep(.4); Path(sys.argv[1]).touch()')
            result = download.run([sys.executable, '-c', script, str(marker)],
                                  timeout=5, emit=emit, log=log, heartbeat=.1)
            self.assertEqual(result, b'\x00archive\xff')
            self.assertTrue(any('50%' in m and not finished for m, finished in messages))
            self.assertNotIn('archive', log.read_text(encoding='utf-8'))
            self.assertIn('Receiving objects', log.read_text(encoding='utf-8'))

    def test_stdout_and_stderr_are_drained_concurrently(self):
        script = ('import os; '
                  '[(os.write(1,b"x"*8192),os.write(2,b"Counting objects: 50% (1/2)\\r"*400)) for _ in range(32)]')
        result = download.run([sys.executable, '-c', script], timeout=10, emit=lambda _: None)
        self.assertEqual(len(result), 8192 * 32)

    def test_heartbeat_does_not_extend_total_timeout(self):
        started = time.monotonic()
        messages = []
        with self.assertRaises(subprocess.TimeoutExpired):
            download.run([sys.executable, '-c', 'import time; time.sleep(30)'],
                         timeout=.3, heartbeat=.05, emit=messages.append)
        self.assertLess(time.monotonic() - started, 5)
        self.assertTrue(any('等待' in m for m in messages))

    def test_diagnostics_distinguish_auth_and_download_disconnect_without_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'download.log'
            for stderr, expected in [('Permission denied (publickey).', '认证'),
                                     ('fatal: early EOF', '下载连接中断'),
                                     ('Host key verification failed.', '主机')]:
                script = 'import sys; sys.stderr.write(sys.argv[1]); sys.exit(1)'
                with self.assertRaises(download.GitFailure) as caught:
                    download.run([sys.executable, '-c', script, stderr + '\nsecret-token\n'],
                                 timeout=5, log=log, emit=lambda _: None)
                self.assertIn(expected, str(caught.exception))
                self.assertNotIn('secret-token', str(caught.exception))
            self.assertNotIn('secret-token', log.read_text(encoding='utf-8'))

    @unittest.skipUnless(os.name == 'posix', 'POSIX process groups')
    def test_timeout_reaps_ssh_like_descendant_that_ignores_term(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'helper.pid'
            helper = ('import os,signal,time,sys; from pathlib import Path; '
                      'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                      'Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)')
            parent = ('import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2]]); time.sleep(30)')
            with self.assertRaises(subprocess.TimeoutExpired):
                download.run([sys.executable, '-c', parent, helper, str(marker)], timeout=1, emit=lambda _: None)
            pid = int(marker.read_text())
            try:
                state = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
            except FileNotFoundError:
                state = 'gone'
            self.assertIn(state, ('gone', 'Z'))


if __name__ == '__main__':
    unittest.main()
