"""顺序运行所选短测，单次硬超时30分钟；失败停止。"""
import subprocess
import sys
import time
import os
from data_pipeline import ROOT, CODE_ROOT, load_json, save_json
from model_catalog import load_selection


def main():
    # 开始前确认两个来源准备都完成，避免后半段失败时误称整个短测完成。
    selected = load_selection(ROOT, default_folds=['ct_holdG'])
    for fold in selected['folds']:
        assert load_json(ROOT / f"09_reports/{fold}_preparation.json")["status"] == "passed"
    suite = ROOT / "09_reports" / f"smoke_suite_{time.strftime('%Y%m%d_%H%M%S')}"
    suite.mkdir(exist_ok=False)
    records = []
    for fold in selected['folds']:
        for method in selected['methods']:
            command = [sys.executable, "-B", "-X", "utf8", str(CODE_ROOT / "05_src/smoke_train.py"), fold, method, "--steps", "12"]
            # 所选B0验证训练、验证、检查点恢复及原图预测。
            if method == selected['methods'][0]:
                command.append("--predict")
            log = suite / f"{fold}_{method}.log"
            started = time.perf_counter()
            record = {"fold": fold, "method": method, "command": command, "hard_timeout_seconds": 1800, "log": str(log), "status": "running"}
            records.append(record)
            save_json(suite / "suite.json", records)
            print(f"START {fold} {method}: {log}", flush=True)
            with log.open("w", encoding="utf-8") as stream:
                try:
                    if os.environ.get('WHOLE_HEART_MANAGED') == '1':
                        from task_worker import run_child
                        result = run_child(command, os.environ.copy(), cwd=ROOT, output=stream, timeout=1800)
                    else:
                        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1800, cwd=ROOT)
                    record.update(status="passed" if result.returncode == 0 else "failed", exit_code=result.returncode)
                except subprocess.TimeoutExpired:
                    record.update(status="timeout", exit_code=None)
                except BaseException as error:
                    record.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),seconds=time.perf_counter()-started,ended=time.strftime('%F %T'))
                    save_json(suite/'suite.json',records)
                    raise
            record["seconds"] = time.perf_counter()-started
            record['ended']=time.strftime('%F %T')
            save_json(suite / "suite.json", records)
            print(f"END {fold} {method}: {record['status']}", flush=True)
            if record["status"] != "passed":
                raise RuntimeError(f"短测停止，请检查日志：{log}")
    print(f"ALL PASSED: {suite}", flush=True)


if __name__ == "__main__":
    main()
