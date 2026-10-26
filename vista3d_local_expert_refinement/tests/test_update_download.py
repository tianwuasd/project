"""Bounded downloads retain actual progress and never switch on failure."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
import task_update as updater


class DownloadTests(unittest.TestCase):
    def test_partial_progress_is_visible_and_logged_before_process_finishes(self):
        with tempfile.TemporaryDirectory() as temp:
            log=Path(temp)/'fetch.log';first=b'Receiving objects: 10%\r';second=first+b'Receiving objects: 50%\r'
            calls=[];output=io.StringIO()
            def communicate(**kwargs):
                calls.append(kwargs)
                if len(calls)==1:raise subprocess.TimeoutExpired(['git'],.01,stderr=first)
                self.assertEqual(log.read_bytes(),first if len(calls)==2 else second)
                self.assertIn('Receiving objects: 10%',output.getvalue())
                if len(calls)==2:raise subprocess.TimeoutExpired(['git'],.01,stderr=second)
                return b'payload',second+b'Receiving objects: 100%\n'
            process=Mock(returncode=0);process.communicate.side_effect=communicate
            with patch.object(updater.subprocess,'Popen',return_value=process),contextlib.redirect_stdout(output):
                result=updater.git(Path(temp),'fetch',timeout=10,binary=True,progress_log=log)
            self.assertEqual(log.read_bytes(),result.stderr)
            self.assertEqual(result.stdout,b'payload')

    def test_interrupt_reaps_owned_child_and_keeps_final_stderr(self):
        with tempfile.TemporaryDirectory() as temp:
            log=Path(temp)/'fetch.log';process=Mock(pid=4321)
            process.poll.return_value=None
            process.communicate.side_effect=[KeyboardInterrupt,(b'',b'last download output\n')]
            with patch.object(updater.subprocess,'Popen',return_value=process),patch.object(updater.os,'killpg',create=True) as killpg,contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):updater.git(Path(temp),'fetch',progress_log=log)
            self.assertEqual(log.read_bytes(),b'last download output\n')
            self.assertEqual(process.communicate.call_count,2)
            if os.name=='posix':killpg.assert_called_once_with(4321,updater.signal.SIGKILL)
            else:process.kill.assert_called_once()

    def command(self, root, script, timeout, log):
        real_popen=subprocess.Popen
        processes=[]
        def spawn(command,**kwargs):
            child=real_popen([sys.executable,'-u','-c',script],**kwargs)
            processes.append(child)
            return child
        with patch.object(updater.subprocess,'Popen',side_effect=spawn), \
             patch.object(updater,'PROGRESS_POLL_SECONDS',.02,create=True), \
             patch.object(updater,'PROGRESS_HEARTBEAT_SECONDS',.01,create=True):
            try:
                return updater.git(root,'fetch',timeout=timeout,binary=True,progress_log=log)
            finally:
                if processes:self.assertTrue(all(p.poll() is not None for p in processes))

    def test_slow_success_streams_progress_without_duplicate_bytes_or_pipe_deadlock(self):
        script="import sys,time;sys.stdout.buffer.write(b'x'*65536);sys.stdout.flush();sys.stderr.buffer.write(b'Receiving objects: 10%\\r'+b'y'*65536);sys.stderr.flush();time.sleep(.12);sys.stderr.write('\\rReceiving objects: 100%\\n');sys.stderr.flush()"
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);log=root/'fetch.log'
            with contextlib.redirect_stdout(io.StringIO()) as output:
                result=self.command(root,script,10,log)
            self.assertEqual(result.stdout,b'x'*65536)
            self.assertEqual(log.read_bytes(),result.stderr)
            self.assertIn(b'Receiving objects: 100%',result.stderr)
            self.assertIn('下载',output.getvalue())

    def test_timeout_keeps_stderr_and_reaps_child(self):
        script="import sys,time;sys.stderr.write('Receiving objects: 12%\\r');sys.stderr.flush();time.sleep(30)"
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);log=root/'fetch.log'
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(subprocess.TimeoutExpired) as caught:
                self.command(root,script,.6,log)
            self.assertEqual(caught.exception.timeout,.6)
            self.assertIn(b'Receiving objects: 12%',log.read_bytes())

    def test_menu_timeout_reports_cause_and_preserves_existing_pointer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);repo=root/'repo';repo.mkdir();runtime=root/'runtime';runtime.mkdir()
            pointer=runtime/'release.json'
            pointer.write_text(json.dumps(dict(repository=str(repo),commit='old',path=str(runtime/'releases/old'))),encoding='utf-8')
            before=pointer.read_bytes();calls=[]
            def fake_git(where,*args,**kwargs):
                calls.append((args,kwargs))
                result=subprocess.CompletedProcess(args,0,stdout='',stderr=b'')
                if args[:2]==('config','--get'):result.stdout=updater.UPSTREAM
                elif args[:2]==('ls-remote','--get-url'):result.stdout=args[2]
                elif args[0]=='ls-remote':result.stdout='new\trefs/heads/main'
                elif args[0]=='archive':result.stdout=b'old source archive'
                elif args[0]=='fetch':raise subprocess.TimeoutExpired(args,kwargs['timeout'])
                return result
            with patch.object(updater,'git',side_effect=fake_git),contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaisesRegex(ValueError,'下载超时.*900') as caught:updater.update(runtime)
            self.assertIn('下载超时',str(caught.exception))
            self.assertEqual(pointer.read_bytes(),before)
            fetch=[(a,k) for a,k in calls if a[0]=='fetch'][0]
            self.assertEqual(fetch[1]['timeout'],900)
            self.assertIn('--progress',fetch[0])
            self.assertIn('下载超时',output.getvalue())
            attempt=json.loads(next((runtime/'update_backups').glob('*/attempts.json')).read_text(encoding='utf-8'))
            self.assertEqual(attempt['official_head'],'new')
            self.assertIn('900',attempt['attempts'][0]['error'])


if __name__=='__main__':unittest.main()
