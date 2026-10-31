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
    def command(self, root, script, timeout, log):
        real_popen=subprocess.Popen;processes=[]
        def spawn(command,**kwargs):
            child=real_popen([sys.executable,'-u','-c',script],**kwargs)
            processes.append(child);return child
        with patch.object(updater.subprocess,'Popen',side_effect=spawn),patch.object(updater,'HEARTBEAT_SECONDS',.05,create=True):
            try:return updater.git(root,'fetch',timeout=timeout,progress_log=log,stage='对象下载')
            finally:
                if processes:self.assertTrue(all(p.poll() is not None for p in processes))

    def test_partial_progress_is_visible_before_exit_and_both_pipes_are_drained(self):
        import time
        script="import sys,time;sys.stdout.write('x'*200000);sys.stdout.flush();sys.stderr.write('Receiving objects: 10% (1/10)\\r');sys.stderr.flush();time.sleep(.4);sys.stderr.write('Receiving objects: 100% (10/10)\\n');sys.stderr.flush()"
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);log=root/'fetch.log';printed=[]
            with patch('builtins.print',side_effect=lambda *a,**k:printed.append((time.monotonic(),' '.join(map(str,a))))):
                result=self.command(root,script,5,log)
            finished=time.monotonic();events=[t for t,line in printed if '10%' in line]
            self.assertTrue(events and events[0]<finished-.2)
            self.assertLessEqual(len(result.stdout),65536)
            self.assertIn('100%',log.read_text(encoding='utf-8'))

    def test_timeout_keeps_stderr_and_reaps_child(self):
        script="import sys,time;sys.stderr.write('Receiving objects: 12%\\r');sys.stderr.flush();time.sleep(30)"
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);log=root/'fetch.log'
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(subprocess.TimeoutExpired) as caught:
                self.command(root,script,.6,log)
            self.assertEqual(caught.exception.timeout,.6)
            self.assertIn('Receiving objects: 12%',log.read_text(encoding='utf-8'))

    def test_cancel_reaps_real_child_and_reader_threads(self):
        import queue,threading
        real_queue=queue.Queue
        class CancelledQueue(real_queue):
            def get(self,**kwargs):raise KeyboardInterrupt
        with tempfile.TemporaryDirectory() as temp,patch.object(updater.queue,'Queue',CancelledQueue):
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(KeyboardInterrupt):
                self.command(Path(temp),'import time;time.sleep(30)',3,Path(temp)/'fetch.log')
        self.assertFalse(any(t.name.startswith('chd-git-') for t in threading.enumerate()))

    def test_connection_failure_is_classified_and_secrets_are_redacted(self):
        script="import sys;sys.stderr.write('Permission denied (publickey).\\nAuthorization: Bearer synthetic-secret\\nhttps://alice:synthetic-secret@example.invalid/repo?token=synthetic-secret\\n');sys.exit(128)"
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);log=root/'fetch.log'
            with self.assertRaisesRegex(ValueError,'对象下载.*认证') as caught:self.command(root,script,3,log)
            self.assertNotIn('synthetic-secret',str(caught.exception)+log.read_text(encoding='utf-8'))
            self.assertIn('Permission denied',caught.exception.output_tail)

    def test_log_is_bounded_but_error_tail_survives(self):
        script="import sys\\nfor i in range(50): sys.stderr.write('remote: '+'x'*4096+'\\r')\\nsys.stderr.write('fatal: Permission denied (publickey).\\n');sys.exit(128)"
        script=script.replace('\\n','\n',2)
        with tempfile.TemporaryDirectory() as temp,patch.object(updater,'MAX_LOG_BYTES',8000,create=True):
            root=Path(temp);log=root/'fetch.log'
            with self.assertRaisesRegex(ValueError,'认证') as caught:self.command(root,script,5,log)
            self.assertLessEqual(log.stat().st_size,8000)
            self.assertIn('Permission denied',caught.exception.output_tail)

    def test_failed_version_query_is_reported_and_persisted_before_fetch_failure(self):
        for failure in [subprocess.TimeoutExpired(['git','ls-remote'],30,stderr=b'Connection timed out during banner exchange\nProxy-Authorization: Bearer synthetic-secret\n'),
                        ValueError('Git failed: Permission denied (publickey)')]:
            with self.subTest(failure=type(failure).__name__),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);repo=root/'repo';repo.mkdir();runtime=root/'runtime';runtime.mkdir()
                pointer=runtime/'release.json'
                pointer.write_text(json.dumps(dict(repository=str(repo),commit='old',path=str(runtime/'releases/old'))),encoding='utf-8')
                before=pointer.read_bytes();calls=[]
                def fake_git(where,*args,**kwargs):
                    calls.append(args)
                    result=subprocess.CompletedProcess(args,0,stdout='',stderr=b'')
                    if args[:2]==('config','--get'):result.stdout=updater.UPSTREAM
                    elif args[:2]==('ls-remote','--get-url'):result.stdout=args[2]
                    elif args[0]=='ls-remote':raise failure
                    elif args[0]=='fetch':raise ValueError('download unavailable')
                    return result
                with patch.object(updater,'git',side_effect=fake_git),contextlib.redirect_stdout(io.StringIO()) as output:
                    with self.assertRaises(ValueError):updater.update(runtime)
                backup=next((runtime/'update_backups').iterdir())
                self.assertTrue((backup/'query.log').is_file(),'Version-query failure was swallowed instead of logged')
                report=json.loads((backup/'attempts.json').read_text(encoding='utf-8'))
                self.assertEqual(report['query']['status'],'failed')
                self.assertEqual(report['query']['log'],str(backup/'query.log'))
                detail=(backup/'query.log').read_text(encoding='utf-8')
                expected='banner exchange' if isinstance(failure,subprocess.TimeoutExpired) else 'Permission denied'
                self.assertIn(expected,detail)
                self.assertIn(expected,output.getvalue())
                self.assertNotIn('synthetic-secret',detail+json.dumps(report)+output.getvalue())
                self.assertEqual(pointer.read_bytes(),before)
                self.assertTrue(any(args[0]=='fetch' for args in calls))

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
                elif args[0]=='ls-remote':result.stdout='a'*40+'\trefs/heads/main'
                elif args[0]=='archive':result.stdout=b'old source archive'
                elif args[0]=='fetch':raise subprocess.TimeoutExpired(args,kwargs['timeout'])
                return result
            with patch.object(updater,'git',side_effect=fake_git),contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaisesRegex(ValueError,'下载.*超时.*900') as caught:updater.update(runtime)
            self.assertIn('总超时',str(caught.exception))
            self.assertEqual(pointer.read_bytes(),before)
            fetch=[(a,k) for a,k in calls if a[0]=='fetch'][0]
            self.assertEqual(fetch[1]['timeout'],900)
            self.assertIn('--progress',fetch[0])
            self.assertIn('总超时',output.getvalue())
            attempt=json.loads(next((runtime/'update_backups').glob('*/attempts.json')).read_text(encoding='utf-8'))
            self.assertEqual(attempt['official_head'],'a'*40)
            self.assertIn('900',attempt['attempts'][0]['error'])


if __name__=='__main__':unittest.main()
