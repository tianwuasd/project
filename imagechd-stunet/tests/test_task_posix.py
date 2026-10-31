"""Run on Linux before deployment; Windows reports explicit skips."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
import task_runtime as jobs


@unittest.skipUnless(sys.platform.startswith('linux'),'requires actual Linux /proc and flock')
class LinuxLifecycleTests(unittest.TestCase):
    def test_child_holds_lock_after_parent_descriptor_closes(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime=Path(temp)
            with jobs.runtime_lock(runtime) as lock:
                process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'],pass_fds=(lock.fileno(),),start_new_session=True)
            try:
                with self.assertRaises(ValueError):
                    with jobs.runtime_lock(runtime):pass
                record={'pid':process.pid,'identity':jobs.identity(process.pid)}
                self.assertTrue(jobs.alive(record))
                self.assertFalse(jobs.alive(dict(record,identity={'boot':'wrong','start':'wrong'})))
            finally:
                process.terminate();process.wait(timeout=5)
            with jobs.runtime_lock(runtime):pass


if __name__=='__main__':unittest.main()
