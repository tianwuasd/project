"""Real child-process progress, timeout, and offline update diagnostics."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import setup_env
from update import UpdateLog


class CLIFeedbackTests(unittest.TestCase):
    def setUp(self):
        folder=ROOT/'outputs/tests-tmp';folder.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(dir=folder)
        self.folder=Path(self.temporary.name)

    def tearDown(self):self.temporary.cleanup()

    def test_child_output_is_visible_before_the_command_exits(self):
        release=self.folder/'release';path=self.folder/'stream.log';console=io.StringIO();result=[]
        code=("import sys,time;from pathlib import Path;"
              "print('中文-output',flush=True);print('stderr-line',file=sys.stderr,flush=True);"
              "p=Path(sys.argv[1]);end=time.monotonic()+10\n"
              "while not p.exists() and time.monotonic()<end:time.sleep(.02)\n")
        with path.open('w',encoding='utf-8') as stream,redirect_stdout(console):
            worker=threading.Thread(target=lambda:result.append(setup_env.run_logged(
                [sys.executable,'-c',code,str(release)],stream,'streaming',timeout=12)))
            worker.start()
            try:
                deadline=time.monotonic()+5
                while 'stderr-line' not in console.getvalue() and time.monotonic()<deadline:time.sleep(.02)
                self.assertIn('中文-output',console.getvalue())
                self.assertIn('stderr-line',console.getvalue())
                self.assertIn('stderr-line',path.read_text(encoding='utf-8'))
                self.assertTrue(worker.is_alive(),'the output must arrive while the child is still running')
            finally:release.touch();worker.join(timeout=6)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result[0].returncode,0)

    def test_silent_child_reports_wait_and_stops_at_timeout(self):
        console=io.StringIO();began=time.monotonic()
        with (self.folder/'timeout.log').open('w',encoding='utf-8') as stream,redirect_stdout(console):
            result=setup_env.run_logged([sys.executable,'-c','import time;time.sleep(10)'],
                stream,'silent',timeout=.4,heartbeat=.08)
        self.assertEqual(result.returncode,124)
        self.assertLess(time.monotonic()-began,5)
        self.assertIn('等待输出',console.getvalue());self.assertIn('超时',console.getvalue())

    def test_failed_install_stops_and_keeps_private_configuration(self):
        local=self.folder/'project.local.json';local.write_text('{"keep":"original"}',encoding='utf-8')
        (self.folder/'requirements-analysis.txt').write_text('example==1',encoding='utf-8')
        result=subprocess.CompletedProcess(['fake'],7,stdout='failed dependency',stderr='')
        with patch.object(setup_env,'ROOT',self.folder),patch.object(sys,'argv',
                ['setup_env.py','--install','--profile','analysis']), \
             patch.object(setup_env,'run_logged',return_value=result) as command,redirect_stdout(io.StringIO()):
            self.assertEqual(setup_env.main(),1)
        self.assertEqual(command.call_count,1)
        self.assertEqual(json.loads(local.read_text()),{'keep':'original'})
        failed=json.loads(next((self.folder/'outputs/environment').glob('*_install_failed.json')).read_text())
        self.assertFalse(failed['passed']);self.assertEqual(failed['steps'][0]['exit_code'],7)

    def test_blocked_update_has_heartbeat_and_failure_receipt(self):
        path=self.folder/'update.log';console=io.StringIO()
        with redirect_stdout(console),self.assertRaisesRegex(ValueError,'bad hash'):
            with UpdateLog(path) as feedback:
                with feedback.waiting('network',interval=.03):time.sleep(.1)
                raise ValueError('bad hash')
        text=path.read_text(encoding='utf-8')
        self.assertIn('等待：network',console.getvalue())
        self.assertIn('Traceback',text)
        record=json.loads(path.with_name('update-result.json').read_text())
        self.assertEqual(record['action'],'failed');self.assertEqual(record['error_type'],'ValueError')


if __name__=='__main__':unittest.main()
