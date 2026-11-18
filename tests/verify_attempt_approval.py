from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from chd_mini.attempts import AttemptLedger, STAGES
from chd_mini.contracts import confined, read_json, write_json


ROOT = Path('D:/projects/back/t17/chd-review-20261118/assistant3-localizer-approval-controls')
REPORT = Path('D:/projects/back/sync-reports/chd-review-20261118/assistant3/localizer-approval-controls.json')
SHARED = Path('D:/projects/back/sync-reports/chd-review-20261118/attempts.json')


def denied(ledger, action, stage='q_base'):
    command = [sys.executable,'-B',__file__,'--child',str(ledger.path),action,stage]
    result = subprocess.run(command,capture_output=True)
    assert result.returncode != 0


def main():
    if len(sys.argv) > 1:
        ledger = AttemptLedger(sys.argv[2])
        ledger.claim(sys.argv[4]) if sys.argv[3] == 'claim' else ledger.read()
        return
    if ROOT.exists():
        raise FileExistsError('独立控制目录已存在')
    confined(ROOT,ROOT.parent)
    shared_before = SHARED.read_bytes()
    ROOT.mkdir(parents=True)
    started = perf_counter()
    results = {}
    try:
        ledger = AttemptLedger(ROOT/'attempts.json')
        ledger.initialize('approval-controls-no-model-training')
        assert ledger.limits() == dict(stage_limits={s:2 for s in STAGES},total_limit=10)
        for stage in ('q_base','q_localizer'):
            for _ in range(2):
                number = ledger.claim(stage)
                ledger.finish(stage,number,'skipped')
            denied(ledger,'claim',stage)
        before = ledger.read()
        denied(ledger,'claim')
        approval = dict(protocol='chd-mini-attempt-approval-v1',run_id=before['run_id'],
            ledger_path=str(ledger.path.resolve()),stage_limits={s:3 if s=='q_base' else 2 for s in STAGES},
            total_limit=11,authorization_date_local='2026-11-18',
            user_reply='独立控制检查：不授权任何模型训练',scope='控制接口边界检查，独立台账')
        sidecar = ledger.path.with_suffix('.approval.json')
        for key, value in (('run_id','wrong-run'),('ledger_path',str(ROOT/'other.json')),
                           ('total_limit',12),('stage_limits',{s:3 for s in STAGES})):
            invalid = deepcopy(approval)
            invalid[key] = value
            write_json(sidecar,invalid)
            denied(ledger,'read')
            denied(ledger,'claim')
        write_json(sidecar,approval)
        assert ledger.read() == before
        assert ledger.limits()['stage_limits']['q_base'] == 3
        number = ledger.claim('q_base')
        assert number == 3
        ledger.finish('q_base',number,'skipped')
        denied(ledger,'claim')
        denied(ledger,'claim','q_localizer')
        previous = ledger.read()
        approval['stage_limits']['q_localizer'] = 3
        approval['total_limit'] = 12
        for key, value in (('run_id','wrong-run'),('ledger_path',str(ROOT/'other.json')),
                           ('total_limit',11),('total_limit',13),('total_limit',True),
                           ('stage_limits',{**approval['stage_limits'],'med_sequence':3}),
                           ('stage_limits',{**approval['stage_limits'],'q_base':2}),
                           ('stage_limits',{**approval['stage_limits'],'stu':2.0}),
                           ('user_reply',''),('unexpected',True)):
            invalid = deepcopy(approval)
            invalid[key] = value
            write_json(sidecar,invalid)
            denied(ledger,'read','q_localizer')
            denied(ledger,'claim','q_localizer')
        write_json(sidecar,approval)
        assert ledger.read() == previous
        assert ledger.limits() == dict(stage_limits=approval['stage_limits'],total_limit=12)
        number = ledger.claim('q_localizer')
        assert number == 3
        ledger.finish('q_localizer',number,'skipped')
        denied(ledger,'claim','q_localizer')
        for stage in ('stu','q_local','med_sequence'):
            for _ in range(2):
                number = ledger.claim(stage)
                ledger.finish(stage,number,'skipped')
            denied(ledger,'claim',stage)
        final = ledger.read()
        assert len(final['attempts']) == 12 and final['attempts'][:len(previous['attempts'])] == previous['attempts']
        assert all(row['outcome']=='skipped' for row in final['attempts'])
        formal = AttemptLedger(SHARED)
        assert len(formal.read()['attempts']) == 10
        assert formal.limits() == dict(stage_limits=approval['stage_limits'],total_limit=12)
        assert SHARED.read_bytes() == shared_before
        results = dict(default_limit=10,previous_approved_limit=11,approved_limit=12,
                       q_base=3,q_localizer=3,other_stages=2,control_claims=12,
                       invalid_approval_read_and_claim_rejected=True,original_records_preserved=True,
                       shared_records_unchanged=True,shared_record_count=10)
    finally:
        rows = list(ROOT.rglob('*'))
        files = [p for p in rows if p.is_file()]
        receipt = dict(results=results,files=len(files),bytes=sum(p.stat().st_size for p in files),
                       wall_seconds=perf_counter()-started,model_training_attempts=0,
                       samples=1,memory_peak='未测量',gpu_memory_peak='未测量')
        confined(ROOT,ROOT.parent)
        shutil.rmtree(ROOT)
        receipt['work_removed'] = not ROOT.exists()
        write_json(REPORT,receipt)
    print(results)


if __name__ == '__main__':
    main()
