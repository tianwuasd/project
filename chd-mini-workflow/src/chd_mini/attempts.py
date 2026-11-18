from pathlib import Path

from filelock import FileLock

from .contracts import read_json, write_json


STAGES = ("stu", "q_base", "q_localizer", "q_local", "med_sequence")
OUTCOMES = ("successful", "skipped", "failed", "interrupted")


class AttemptLedger:
    def __init__(self, path):
        self.path = Path(path)

    def _limits(self, state):
        defaults = dict(stage_limits={stage: 2 for stage in STAGES}, total_limit=10)
        approval_path = self.path.with_suffix('.approval.json')
        if not approval_path.exists():
            return defaults
        approval = read_json(approval_path)
        fields = {'protocol','run_id','ledger_path','stage_limits','total_limit',
                  'authorization_date_local','user_reply','scope'}
        if set(approval) != fields or approval['protocol'] != 'chd-mini-attempt-approval-v1':
            raise ValueError('尝试批准文件协议或字段无效')
        if approval['run_id'] != state['run_id'] or Path(approval['ledger_path']).resolve() != self.path.resolve():
            raise ValueError('尝试批准文件与本轮台账身份不符')
        if any(not isinstance(approval[k],str) or not approval[k].strip()
               for k in ('run_id','ledger_path','authorization_date_local','user_reply','scope')):
            raise ValueError('尝试批准文件缺少明确授权记录')
        approved = approval['stage_limits']
        expected = {**defaults['stage_limits'], 'q_base': 3}
        approved_options = ((expected, 11), ({**expected, 'q_localizer': 3}, 12))
        if (not isinstance(approved,dict) or set(approved) != set(STAGES) or
                any(type(n) is not int for n in approved.values()) or
                type(approval['total_limit']) is not int or
                (approved, approval['total_limit']) not in approved_options):
            raise ValueError('批准仅允许Q基础模型3次及可选Q定位器3次，其余阶段2次，总计分别11或12次')
        return dict(stage_limits=dict(approved), total_limit=approval['total_limit'])

    def limits(self):
        state = self.read()
        return self._limits(state)

    def initialize(self, run_id):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.path) + ".lock", timeout=0):
            if self.path.exists():
                state = self.read()
                if state["run_id"] != run_id:
                    raise ValueError("尝试记录属于其他运行")
            else:
                state = {"protocol": "chd-mini-attempt-v1", "run_id": run_id, "attempts": []}
                self._limits(state)
                write_json(self.path, state)

    def read(self):
        state = read_json(self.path)
        if state["protocol"] != "chd-mini-attempt-v1":
            raise ValueError("尝试记录协议不受支持")
        limits = self._limits(state)
        counts = {stage: 0 for stage in STAGES}
        for row in state["attempts"]:
            counts[row["stage"]] += 1
            if row["number"] != counts[row["stage"]] or row["outcome"] not in (*OUTCOMES, "started"):
                raise ValueError("尝试记录次序或状态无效")
        if (len(state["attempts"]) > limits['total_limit'] or
                any(n > limits['stage_limits'][stage] for stage,n in counts.items())):
            raise ValueError("尝试记录超过批准上限")
        return state

    def claim(self, stage):
        if stage not in STAGES:
            raise ValueError("未知训练阶段")
        with FileLock(str(self.path) + ".lock", timeout=0):
            state = self.read()
            limits = self._limits(state)
            rows = [row for row in state["attempts"] if row["stage"] == stage]
            if any(row["outcome"] == "started" for row in state["attempts"]):
                raise RuntimeError("存在未结算尝试，须确认进程退出并登记中断")
            if len(rows) >= limits['stage_limits'][stage] or len(state["attempts"]) >= limits['total_limit']:
                raise RuntimeError("本轮累计尝试额度已经耗尽")
            number = len(rows) + 1
            state["attempts"].append({"stage": stage, "number": number, "outcome": "started"})
            write_json(self.path, state)
            return number

    def finish(self, stage, number, outcome):
        if type(number) is not int or number < 1 or stage not in STAGES or outcome not in OUTCOMES:
            raise ValueError("未知尝试结果")
        with FileLock(str(self.path) + ".lock", timeout=0):
            state = self.read()
            rows = [r for r in state["attempts"] if r["stage"] == stage and r["number"] == number]
            if len(rows) != 1 or rows[0]["outcome"] != "started":
                raise ValueError("尝试缺失或已经结算")
            rows[0]["outcome"] = outcome
            write_json(self.path, state)
        return state
