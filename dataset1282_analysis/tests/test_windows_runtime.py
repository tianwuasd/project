from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from dataset1282_analysis.contracts import atomic_json,read_json
from dataset1282_analysis.observability.status import process_birth


@unittest.skipUnless(os.name=='nt','Windows 文件与进程句柄语义')
class WindowsRuntimeTests(unittest.TestCase):
    def test_exited_process_has_no_live_identity_while_handle_is_retained(self):
        process = subprocess.Popen([sys.executable,'-c','pass'])
        process.wait(timeout=10)
        self.assertIsNone(process_birth(process.pid))

    def test_atomic_state_waits_for_brief_reader_then_replaces(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/'state.json'
            atomic_json(file,{'state':'old'})
            with ThreadPoolExecutor(max_workers=1) as pool:
                with file.open('rb'):
                    update = pool.submit(atomic_json,file,{'state':'new'})
                    time.sleep(.15)
                update.result(timeout=5)
            self.assertEqual(read_json(file),{'state':'new'})
