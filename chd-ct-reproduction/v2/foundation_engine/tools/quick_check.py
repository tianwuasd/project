"""Short, offline engineering check. No installation, server job or real data."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest


def main():
    root=Path(__file__).resolve().parents[1]
    syntax=0
    for path in sorted((root/'05_src').rglob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path));syntax+=1
    sys.path[:0]=[str(root/'05_src'),str(root/'tests')]
    suite=unittest.TestSuite()
    for pattern in ['test_capped_batch.py','test_performance_repair.py','test_training_policy.py','test_memory_guard.py',
                    'test_live_resources.py','test_install_routes.py','test_gpu_memory.py']:
        suite.addTests(unittest.defaultTestLoader.discover(str(root/'tests'),pattern=pattern))
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
        result=unittest.TextTestRunner(stream=capture,verbosity=1).run(suite)
    summary=dict(status='passed' if result.wasSuccessful() else 'failed',syntax_files=syntax,
        tests=result.testsRun,skipped=len(result.skipped),failures=len(result.failures),errors=len(result.errors),
        scope='offline control checks; no real CT, CUDA/NCCL or sustained throughput validation')
    if not result.wasSuccessful():summary['details']=capture.getvalue()[-6000:]
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
