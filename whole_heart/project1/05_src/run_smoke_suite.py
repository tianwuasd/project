"""顺序运行四个短测，单次硬超时 30 分钟；失败即停止，不自动重试。"""
import subprocess
import sys
import time
from data_pipeline import ROOT, load_json, save_json


def main():
    # 开始前确认两个来源准备都完成，避免后半段失败时误称整个短测完成。
    for fold in ["ct_holdG", "mr_holdE"]:
        assert load_json(ROOT / f"09_reports/{fold}_preparation.json")["status"] == "passed"
    suite = ROOT / "09_reports" / f"smoke_suite_{time.strftime('%Y%m%d_%H%M%S')}"
    suite.mkdir(exist_ok=False)
    records = []
    for fold in ["ct_holdG", "mr_holdE"]:
        for method in ["B0", "B1"]:
            command = [sys.executable, "-B", "-X", "utf8", str(ROOT / "05_src/smoke_train.py"), fold, method, "--steps", "12"]
            # 每种模态只做一次完整预测，B1 仍验证训练、开发集和检查点恢复。
            if method == "B0":
                command.append("--predict")
            log = suite / f"{fold}_{method}.log"
            started = time.perf_counter()
            record = {"fold": fold, "method": method, "command": command, "hard_timeout_seconds": 1800, "log": str(log), "status": "running"}
            records.append(record)
            save_json(suite / "suite.json", records)
            print(f"START {fold} {method}: {log}", flush=True)
            with log.open("w", encoding="utf-8") as stream:
                try:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1800, cwd=ROOT)
                    record.update(status="passed" if result.returncode == 0 else "failed", exit_code=result.returncode)
                except subprocess.TimeoutExpired:
                    record.update(status="timeout", exit_code=None)
            record["seconds"] = time.perf_counter()-started
            save_json(suite / "suite.json", records)
            print(f"END {fold} {method}: {record['status']}", flush=True)
            if record["status"] != "passed":
                raise RuntimeError(f"短测停止，请检查日志：{log}")
    print(f"ALL PASSED: {suite}", flush=True)


if __name__ == "__main__":
    main()
