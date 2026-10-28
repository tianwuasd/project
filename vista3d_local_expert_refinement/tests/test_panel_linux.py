"""Real Linux process and Bash contracts with inert fake dependency commands."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'05_src'))
from server_data import write_json, read_json
from task_runtime import identity
from task_progress import process_lines

CODE = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform.startswith('linux'), 'requires actual Linux /proc, Bash and flock')
class LinuxPanelTests(unittest.TestCase):
    def test_verified_process_tree_includes_child_and_excludes_unrelated(self):
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp)
            script = 'import subprocess,sys,time; c=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);print(c.pid,flush=True);time.sleep(30)'
            parent = subprocess.Popen([sys.executable,'-c',script], stdout=subprocess.PIPE, text=True, start_new_session=True)
            unrelated = subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
            try:
                child = int(parent.stdout.readline().strip())
                write_json(job/'state.json', dict(pid=parent.pid, identity=identity(parent.pid), role='worker'))
                text = '\n'.join(process_lines(job))
                self.assertIn(f'{parent.pid} /',text)
                self.assertIn(f'{child} /',text)
                self.assertNotIn(f'{unrelated.pid} /',text)
                self.assertIn('CPU累计秒',text)
                write_json(job/'state.json',dict(pid=parent.pid, identity={'boot':'wrong','start':'wrong'}))
                self.assertIn('没有身份匹配', '\n'.join(process_lines(job)))
            finally:
                os.killpg(parent.pid,signal.SIGTERM);parent.wait(timeout=5)
                parent.stdout.close()
                unrelated.terminate();unrelated.wait(timeout=5)

    def fixture(self, root):
        conda = root/'fake-conda'
        conda.write_text('''#!/bin/bash
set -eu
printf '%s\\n' "$*" >> "$CALLS"
if [[ "$1" == config ]]; then
  "$WHOLE_HEART_CONTROL_PYTHON" -c 'import json,os; print(json.dumps({"proxy_servers":json.load(open(os.environ["CONDARC"]))["proxy_servers"]}))'
fi
if [[ "$1" == env ]]; then
  while [[ "$1" != --prefix ]]; do shift; done
  mkdir -p "$2/bin"
  touch "$2/bin/python"; chmod +x "$2/bin/python"
fi
if [[ "${FAKE_FAIL_CHECK:-0}" == 1 && "$*" == *'pip check'* ]]; then exit 4; fi
if [[ "${FAKE_FAIL_IMPORT_ONCE:-0}" == 1 && "$*" == *'--check-dependencies'* && ! -f "$CALLS.import-failed" ]]; then
  touch "$CALLS.import-failed"; echo "No module named psutil" >&2; exit 5
fi
if [[ "${FAKE_FAIL_IMPORT_ALWAYS:-0}" == 1 && "$*" == *'--check-dependencies'* ]]; then exit 5; fi
''', encoding='utf-8')
        conda.chmod(0o700)
        env = dict(os.environ, CONDA_EXE=str(conda), WHOLE_HEART_ENV_PREFIX=str(root/'env'),
                   WHOLE_HEART_JOB=str(root/'job'), WHOLE_HEART_CONTROL_PYTHON=sys.executable,
                   CALLS=str(root/'calls'))
        return env

    def run_setup(self, env):
        return subprocess.run(['bash',str(CODE/'bootstrap_server.sh'),'--install-only'],env=env,
                              text=True,capture_output=True,timeout=30)

    def test_installer_signed_marker_reuse_and_legacy_marker_upgrade(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);env=self.fixture(root)
            result=self.run_setup(env)
            self.assertEqual(result.returncode,0,result.stderr)
            marker=root/'env/.baseline_ready'
            self.assertEqual(len(marker.read_text().strip()),64)
            self.assertEqual(read_json(root/'job/environment.json')['status'],'completed')
            first=(root/'calls').read_text().count('pip install')
            result=self.run_setup(env)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual((root/'calls').read_text().count('pip install'),first)
            marker.write_text('')
            result=self.run_setup(env)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertGreater((root/'calls').read_text().count('pip install'),first)

    def test_failed_dependency_check_never_marks_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);env=self.fixture(root)
            result=self.run_setup(dict(env,FAKE_FAIL_CHECK='1'))
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((root/'env/.baseline_ready').exists())
            self.assertEqual(read_json(root/'job/environment.json')['status'],'failed')
            self.assertEqual(read_json(root/'job/environment.json')['phase'],'verify')

    def test_matching_marker_does_not_hide_missing_import_and_repairs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);env=self.fixture(root)
            self.assertEqual(self.run_setup(env).returncode,0)
            first=(root/'calls').read_text().count('pip install')
            result=self.run_setup(dict(env,FAKE_FAIL_IMPORT_ONCE='1'))
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertGreater((root/'calls').read_text().count('pip install'),first)
            self.assertEqual(read_json(root/'job/environment.json')['status'],'completed')

    def test_broken_import_blocks_installation_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);env=self.fixture(root)
            result=self.run_setup(dict(env,FAKE_FAIL_IMPORT_ALWAYS='1'))
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((root/'env/.baseline_ready').exists())
            self.assertEqual(read_json(root/'job/environment.json')['status'],'failed')


if __name__ == '__main__': unittest.main()
