import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from chd_mini.attempts import AttemptLedger
from chd_mini.contracts import read_json, write_json
from chd_mini.review import ReviewFinalizer
from chd_mini.windows_job import supervise


BASE = Path('D:/projects/back/t17/chd-review-20261118/assistant3-finalization-controls')
REPORT = Path('D:/projects/back/sync-reports/chd-review-20261118/assistant3/finalization-controls')


def worker(mode):
    root, reports = BASE/mode, REPORT/mode
    reports.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True)
    ledger = AttemptLedger(reports/'controls-ledger.json')
    ledger.initialize('finalization-controls-no-model-training')
    handoff = dict(work_root=str(BASE), report_root=str(REPORT), attempt_ledger=str(ledger.path))
    plan = dict(engine_commit='control-only', ledger=ledger.read())
    process_paths = []
    with ReviewFinalizer(handoff, plan, root, reports, [], process_paths):
        (root/'owned.txt').write_text('actual owned control artifact', encoding='utf-8')
        if mode == 'before_process':
            raise RuntimeError('控制检查：进程启动前的真实异常路径')
        if mode == 'capacity_error':
            # 不制造超额产物；最终容量读取非法路径仍须执行原目录清理。
            handoff['report_root'] = None
            raise RuntimeError('控制检查：容量路径错误')
        process_path = reports/'failure.process.json'
        process_paths.append(process_path)
        supervise([sys.executable, '-B', '-c', 'import sys;sys.exit(3)'], root, dict(os.environ),
                  reports/'failure.log', 10, 65536, 256*1024**2, process_path)


def main():
    if BASE.exists():
        raise FileExistsError('控制检查目录已存在')
    started = perf_counter()
    results = {}
    for mode in ('before_process', 'capacity_error', 'process_failure'):
        child = subprocess.run([sys.executable, '-B', __file__, '--worker', mode], capture_output=True)
        assert child.returncode != 0
        cleanup = read_json(REPORT/mode/'cleanup.json')
        assert cleanup['state'] == 'completed' and not (BASE/mode).exists()
        results[mode] = dict(exit_code=child.returncode, cleanup=cleanup)
    BASE.rmdir()
    write_json(REPORT/'summary.json', dict(results=results, wall_seconds=perf_counter()-started,
               model_training_attempts=0, note='真实文件与非模型失败进程控制检查'))
    print(results)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        worker(sys.argv[2])
    else:
        main()
