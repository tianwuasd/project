from pathlib import Path

from filelock import FileLock

from .contracts import read_json, write_json


STAGES = ("stu", "q_base", "q_localizer", "q_local", "med_sequence")
OUTCOMES = ("successful", "skipped", "failed", "interrupted")


class AttemptLedger:
    def __init__(self, path):
        self.path = Path(path)

    def initialize(self, run_id):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.path) + ".lock", timeout=0):
            if self.path.exists():
                state = self.read()
                if state["run_id"] != run_id:
                    raise ValueError("尝试记录属于其他运行")
            else:
                write_json(self.path, {"protocol": "chd-mini-attempt-v1", "run_id": run_id,
                                       "attempts": []})

    def read(self):
        state = read_json(self.path)
        if state["protocol"] != "chd-mini-attempt-v1":
            raise ValueError("尝试记录协议不受支持")
        counts = {stage: 0 for stage in STAGES}
        for row in state["attempts"]:
            counts[row["stage"]] += 1
            if row["number"] != counts[row["stage"]] or row["outcome"] not in (*OUTCOMES, "started"):
                raise ValueError("尝试记录次序或状态无效")
        if len(state["attempts"]) > 10 or any(n > 2 for n in counts.values()):
            raise ValueError("尝试记录超过批准上限")
        return state

    def claim(self, stage):
        if stage not in STAGES:
            raise ValueError("未知训练阶段")
        with FileLock(str(self.path) + ".lock", timeout=0):
            state = self.read()
            rows = [row for row in state["attempts"] if row["stage"] == stage]
            if any(row["outcome"] == "started" for row in state["attempts"]):
                raise RuntimeError("存在未结算尝试，须确认进程退出并登记中断")
            if len(rows) >= 2 or len(state["attempts"]) >= 10:
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
