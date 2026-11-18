import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from chd_mini.attempts import AttemptLedger, STAGES
from chd_mini.contracts import confined, write_json
from chd_mini.windows_job import supervise


def main():
    base = Path('D:/projects/back/t17/chd-review-20261118/assistant3-controls')
    reports = Path('D:/projects/back/sync-reports/chd-review-20261118/assistant3')
    if base.exists():
        raise FileExistsError('检查目录已经存在，须核对归属')
    base.mkdir(parents=True)
    reports.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    results = {}
    try:
        ledger = AttemptLedger(base/'controls-ledger.json')
        ledger.initialize('controls-only-no-model-training')
        for stage in STAGES:
            for _ in range(2):
                number = ledger.claim(stage)
                ledger.finish(stage, number, 'skipped')
        state = ledger.read()
        assert len(state['attempts']) == 10
        assert not any(r['outcome'] == 'successful' for r in state['attempts'])
        tool = Path(__file__).resolve().parents[1]/'run.py'
        rejected = subprocess.run([sys.executable, '-B', str(tool), 'attempt', '--ledger', str(ledger.path),
                                   '--stage', 'stu'], capture_output=True)
        assert rejected.returncode != 0 and ledger.read() == state
        ledger.initialize('controls-only-no-model-training')
        assert ledger.read() == state
        results['ledger'] = '10次控制接口调用，全部skipped；与真实训练唯一台账独立，不是模型证据'
        # 主进程自然退出后，其仍在等待的真实子进程必须由Job终止。
        code = "import subprocess,sys;subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)']);print('root-exited')"
        result = supervise([sys.executable, '-B', '-c', code], base, dict(os.environ),
                           base/'process.log', 10, 65536, 256*1024**2, base/'process.json')
        assert result['all_workers_exited'] and result['exit_code'] == 0
        results['windows_job'] = result
    finally:
        rows = [p for p in base.rglob('*') if p.is_file()]
        receipt = dict(results=results, files=len(rows), bytes=sum(p.stat().st_size for p in rows),
                       wall_seconds=perf_counter()-started, model_training_attempts=0)
        confined(base, base.parent)
        shutil.rmtree(base)
        receipt['work_removed'] = not base.exists()
        write_json(reports/'controls.json', receipt)
    print(results)


if __name__ == '__main__':
    main()
